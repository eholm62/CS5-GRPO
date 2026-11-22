# grpo.py
"""
From-scratch GRPO (Group Relative Policy Optimization) implementation.

Implements GRPO as described in the DeepSeekMath paper (Section 4),
with an interface matching TRL's GRPOTrainer.

L = -|T|^{-1} sum min(rho*A, clip(rho,1-eps,1+eps)*A)
    + beta * |T|^{-1} sum (log pi - log pi_ref)
over completion-token positions t, with group-normalised advantages.
"""

import copy
import inspect
import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.utils import is_peft_available

logger = logging.getLogger(__name__)


def _first_attr(obj, names, default=None):
    for name in names:
        v = getattr(obj, name, None)
        if v is not None:
            return v
    return default


class GRPOTrainer(Trainer):
    """GRPO trainer matching TRL's GRPOTrainer calling interface."""

    def __init__(
        self,
        model,
        reward_funcs,
        args,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        tokenizer=None,
        reward_processing_classes=None,
        train_collate_fn=None,
        eval_collate_fn=None,
        peft_config=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
        **kwargs,
    ):
        self.reward_funcs = (
            reward_funcs if isinstance(reward_funcs, (list, tuple))
            else [reward_funcs]
        )
        self.reward_processing_classes = reward_processing_classes
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn
        self.peft_config = peft_config
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self._pending_metrics: dict = {}

        if args is not None:
            args.remove_unused_columns = False

        # PEFT
        self._is_peft = peft_config is not None
        if self._is_peft:
            if not is_peft_available():
                raise ImportError("peft must be installed for peft_config.")
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)

        # Processing class: use explicit arg, else infer from model
        if processing_class is not None:
            self.processing_class = processing_class
        elif tokenizer is not None:
            self.processing_class = tokenizer
        else:
            self.processing_class = self._infer_processing_class(model)

        pad = getattr(self.processing_class, "pad_token_id", None)
        self._pad_token_id = pad if pad is not None else (
            getattr(self.processing_class, "eos_token_id", 0) or 0
        )

        # Reference model
        self.ref_model = None
        beta = float(getattr(args, "beta", 0.0) or 0.0)
        if beta > 0.0 and not self._is_peft:
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        # Delegate to Trainer
        if optimizers is None:
            optimizers = (None, None)
        trainer_kwargs: dict = dict(
            model=model, args=args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=compute_metrics, callbacks=callbacks,
            optimizers=optimizers,
        )
        t_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in t_params:
            trainer_kwargs["processing_class"] = self.processing_class
        elif "tokenizer" in t_params:
            trainer_kwargs["tokenizer"] = self.processing_class
        if "preprocess_logits_for_metrics" in t_params:
            trainer_kwargs["preprocess_logits_for_metrics"] = (
                preprocess_logits_for_metrics
            )
        super().__init__(**trainer_kwargs)

        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.model.device)

    def _infer_processing_class(self, model):
        """Infer a tokenizer/processor from the model's pretrained name."""
        model_name = None
        if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        elif hasattr(model, "name_or_path"):
            model_name = model.name_or_path

        if model_name is None:
            raise ValueError(
                "Cannot infer a processing_class (tokenizer) from the model. "
                "Please pass processing_class or tokenizer explicitly."
            )

        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)

    # -- Data loading ------------------------------------------------------

    @staticmethod
    def _prompt_collate(examples):
        return {"prompts": [ex["prompt"] for ex in examples]}

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        seed = _first_attr(self.args, ("data_seed", "seed"), 42)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._prompt_collate,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            generator=torch.Generator().manual_seed(int(seed)),
        )

    def get_eval_dataloader(self, eval_dataset=None):
        data = eval_dataset if eval_dataset is not None else self.eval_dataset
        if data is None:
            return None
        return DataLoader(
            data,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self._prompt_collate,
        )

    # -- Loss computation (called by default Trainer.training_step) --------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = self._grpo_forward(model, inputs["prompts"])
        if return_outputs:
            return result["loss"], result
        return result["loss"]

    def compute_loss_and_metrics(self, model, inputs, num_items_in_batch=None):
        """Public alias matching TRL's interface."""
        return self.compute_loss(model, inputs, return_outputs=True,
                                 num_items_in_batch=num_items_in_batch)

    def log(self, logs, start_time=None, **kwargs):
        if self._pending_metrics:
            for k, v in self._pending_metrics.items():
                logs.setdefault(k, v)
            self._pending_metrics = {}
        return super().log(logs, start_time, **kwargs)

    # -- Sampling ----------------------------------------------------------

    def _max_prompt_length(self):
        v = _first_attr(self.args, ("max_prompt_length",))
        if v is not None:
            return int(v)
        total = _first_attr(self.args, ("max_length",))
        mcl = _first_attr(self.args, ("max_completion_length",))
        if total is not None and mcl is not None:
            return max(int(total) - int(mcl), 1)
        return 128

    def _max_completion_length(self):
        v = _first_attr(self.args, ("max_completion_length",))
        if v is not None:
            return int(v)
        total = _first_attr(self.args, ("max_length",))
        if total is not None:
            return max(int(total) - self._max_prompt_length(), 1)
        return 32

    @torch.no_grad()
    def _sample_completions(self, model, flat_prompts):
        tok = self.processing_class
        temperature = float(getattr(self.args, "temperature", 1.0) or 1.0)
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self._max_completion_length(),
            do_sample=do_sample,
            pad_token_id=self._pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = float(getattr(self.args, "top_p", 1.0) or 1.0)
            top_k = getattr(self.args, "top_k", 0)
            if top_k and int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)

        was_training = model.training
        model.eval()
        try:
            pids_list, cids_list = [], []
            for p in flat_prompts:
                enc = tok(p, truncation=True,
                          max_length=self._max_prompt_length(),
                          return_tensors="pt")
                ids = enc["input_ids"].to(model.device)
                am = enc["attention_mask"].to(model.device)
                out = model.generate(input_ids=ids, attention_mask=am, **gen_kwargs)
                pids_list.append(ids[0].cpu())
                cids_list.append(out[0, ids.shape[1]:].cpu())
        finally:
            if was_training:
                model.train()
        return pids_list, cids_list

    # -- Batch construction ------------------------------------------------

    def _build_training_batch(self, pids_list, cids_list):
        pad = self._pad_token_id
        seqs, cms = [], []
        for p, c in zip(pids_list, cids_list):
            seqs.append(torch.cat([p, c]))
            cms.append(torch.cat([
                torch.zeros(len(p), dtype=torch.long),
                torch.ones(len(c), dtype=torch.long),
            ]))
        L = max(s.shape[0] for s in seqs)
        B = len(seqs)
        input_ids = torch.full((B, L), pad, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        cmask = torch.zeros((B, L), dtype=torch.long)
        for i, (s, m) in enumerate(zip(seqs, cms)):
            sl = s.shape[0]
            input_ids[i, :sl] = s
            attn[i, :sl] = 1
            cmask[i, :sl] = m
        return input_ids, attn, cmask

    # -- Log-prob helpers --------------------------------------------------

    @staticmethod
    def _token_logps(logits, input_ids):
        sl = logits[:, :-1, :]
        st = input_ids[:, 1:]
        lp = F.log_softmax(sl, dim=-1)
        return torch.gather(lp, -1, st.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _token_entropy(logits):
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)
        return -(lp.exp() * lp).sum(-1)

    # -- Reward & advantage ------------------------------------------------

    def _compute_rewards(self, flat_prompts, cids_list):
        proc = (self.reward_processing_classes[-1]
                if self.reward_processing_classes
                else self.processing_class)
        completions = [proc.decode(c, skip_special_tokens=False) for c in cids_list]
        cids = [c.tolist() for c in cids_list]
        all_scores = [
            rf(prompts=flat_prompts, completions=completions,
               completion_ids=cids)
            for rf in self.reward_funcs
        ]
        if len(all_scores) == 1:
            return [float(s) for s in all_scores[0]]
        return [float(sum(v)) for v in zip(*all_scores)]

    def _group_advantages(self, rewards):
        G = int(self.args.num_generations)
        P = len(rewards) // G
        r = torch.tensor(rewards, dtype=torch.float32).view(P, G)
        mean = r.mean(1, keepdim=True)
        std = r.std(1, keepdim=True, unbiased=False)
        self._last_reward_std = std.mean()
        return ((r - mean) / (std + 1e-4)).view(-1)

    # -- Core GRPO forward -------------------------------------------------

    def _grpo_forward(self, model, prompts):
        args = self.args

        G_raw = getattr(args, "num_generations", None)
        if G_raw is None:
            raise ValueError(
                "args.num_generations is required for GRPO."
            )
        G = int(G_raw)
        if G < 1:
            raise ValueError(f"num_generations must be >= 1, got {G}.")

        device = model.device
        flat_prompts = [p for p in prompts for _ in range(G)]

        # Sample
        pids, cids = self._sample_completions(model, flat_prompts)
        input_ids, attn, cmask = self._build_training_batch(pids, cids)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        cmask = cmask.to(device)

        shift_mask = cmask[:, 1:].float()
        denom = shift_mask.sum().clamp(min=1.0)

        # Current logps (with grad)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits.float()
        cur_logps = self._token_logps(logits, input_ids)

        # Old logps (single-step: detached current)
        old_logps = cur_logps.detach()

        # Ref logps
        if self.ref_model is not None:
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=input_ids, attention_mask=attn, use_cache=False
                )
                ref_logps = self._token_logps(ref_out.logits.float(), input_ids)
        elif self._is_peft and hasattr(model, "disable_adapter"):
            with model.disable_adapter(), torch.no_grad():
                ref_out = model(
                    input_ids=input_ids, attention_mask=attn, use_cache=False
                )
                ref_logps = self._token_logps(ref_out.logits.float(), input_ids)
        else:
            ref_logps = None

        # Ratio
        ratio = torch.exp(cur_logps - old_logps)

        # KL per token
        if ref_logps is not None:
            kl_tok = cur_logps - ref_logps
        else:
            kl_tok = torch.zeros_like(cur_logps)

        # Rewards & advantages
        rewards = self._compute_rewards(flat_prompts, cids)
        adv = self._group_advantages(rewards).to(device)
        reward_mean = torch.tensor(rewards, dtype=torch.float32).mean()

        # PPO clipped surrogate
        eps = float(_first_attr(args, ("epsilon", "epsilon_low"), 0.2) or 0.2)
        adv_tok = adv.unsqueeze(1)
        surr1 = ratio * adv_tok
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_tok
        pg_loss = -torch.min(surr1, surr2).sum() / denom

        # KL penalty
        kl_loss = (kl_tok * shift_mask).sum() / denom

        # Total
        beta = float(getattr(args, "beta", 0.0) or 0.0)
        loss = pg_loss + beta * kl_loss

        # Metrics (detached logits for entropy)
        clip_frac = ((ratio - 1.0).abs() > eps).float().mul(shift_mask).sum() / denom
        entropy = (self._token_entropy(logits.detach()) * shift_mask).sum() / denom
        comp_len = cmask.sum(1).float().mean()

        metrics = {
            "loss": loss.detach(),
            "reward": reward_mean.detach(),
            "reward_std": self._last_reward_std.detach(),
            "kl": kl_loss.detach(),
            "entropy": entropy.detach(),
            "clip_ratio": clip_frac.detach(),
            "completion_length": comp_len.detach(),
        }
        self._pending_metrics = {k: float(v.item()) for k, v in metrics.items()}
        return {"loss": loss, "metrics": metrics}