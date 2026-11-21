"""From-scratch GRPO (Group Relative Policy Optimization) trainer.

Mirrors the public interface and the training behavior of TRL's GRPOTrainer.
It is a transformers.Trainer subclass (like TRL's) so the standard HF training
machinery -- optimizer, LR schedule, logging, .train_loss / .state.log_history --
is present and behaves the same as TRL's. The GRPO-specific parts (completion
sampling, reward scoring, group-normalized advantages, PPO-clipped policy loss
and KL-to-reference penalty) are implemented from scratch below.
"""

import copy
import inspect
import logging
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.utils import is_peft_available

logger = logging.getLogger(__name__)


class GRPOTrainer(Trainer):
    """A from-scratch GRPO trainer matching TRL's GRPOTrainer interface."""

    def __init__(
        self,
        model,
        reward_funcs,
        args,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_class=None,
        train_collate_fn=None,
        eval_collate_fn=None,
        peft_config=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=None,
        preprocess_logits_for_metrics=None,
    ):
        self.reward_funcs = reward_funcs
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn
        self.peft_config = peft_config
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self._pending_metrics = {}

        # The dataset column 'prompt' is not in model.forward's signature; keep it.
        if args is not None:
            args.remove_unused_columns = False

        # Optionally wrap the model with PEFT (mirrors TRL).
        self._is_peft = peft_config is not None
        if self._is_peft:
            if is_peft_available():
                from peft import get_peft_model

                model = get_peft_model(model, peft_config)
            else:
                raise ImportError("peft is required when a peft_config is provided")

        # Resolve the processing class (tokenizer).
        if processing_class is None:
            processing_class = getattr(model, "processing_class", None) or getattr(model, "tokenizer", None)
        self.processing_class = processing_class
        self.reward_processing_class = reward_processing_class or processing_class
        if processing_class is not None:
            pad = processing_class.pad_token_id
            self._pad_token_id = pad if pad is not None else (processing_class.eos_token_id or 0)
        else:
            self._pad_token_id = 0

        # A frozen reference policy for the KL penalty (only when beta > 0).
        self.ref_model = None
        if args is not None and float(getattr(args, "beta", 0.0)) > 0.0 and not self._is_peft:
            self.ref_model = self._build_reference_model(model)

        # Delegate the standard setup to transformers.Trainer (same path TRL uses).
        optimizers = tuple(optimizers) if optimizers is not None else (None, None)
        trainer_kwargs = dict(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        t_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in t_params:
            trainer_kwargs["processing_class"] = processing_class
        elif "tokenizer" in t_params:
            trainer_kwargs["tokenizer"] = processing_class
        if "preprocess_logits_for_metrics" in t_params:
            trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics
        super().__init__(**trainer_kwargs)

        # Re-seat the reference model on the correct device, frozen, after Trainer setup.
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self.model.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Data pipeline: batches of raw prompt strings (not model inputs)
    # ------------------------------------------------------------------ #
    def _prompt_collate_fn(self, examples):
        return {"prompts": [ex["prompt"] for ex in examples]}

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        generator = torch.Generator().manual_seed(self.args.seed)
        params = dict(
            batch_size=int(self.args.per_device_train_batch_size),
            shuffle=bool(self.args.shuffle),
            collate_fn=self._prompt_collate_fn,
            num_workers=int(getattr(self.args, "dataloader_num_workers", 0)),
            pin_memory=bool(getattr(self.args, "dataloader_pin_memory", False)),
            drop_last=bool(self.args.dataloader_drop_last),
            generator=generator,
        )
        return DataLoader(self.train_dataset, **params)

    def get_eval_dataloader(self, eval_dataset=None):
        data = eval_dataset if eval_dataset is not None else self.eval_dataset
        if data is None:
            return None
        return DataLoader(
            data,
            batch_size=int(self.args.per_device_train_batch_size),
            collate_fn=self._prompt_collate_fn,
        )

    # ------------------------------------------------------------------ #
    # Training step + loss (called by Trainer.train loop)
    # ------------------------------------------------------------------ #
    def training_step(self, model, inputs):
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = self._grpo_forward_and_metrics(model, inputs["prompts"])
        if return_outputs:
            return out["loss"], out
        return out["loss"]

    def log(self, logs, start_time=None, **kwargs):
        if getattr(self, "_pending_metrics", None):
            for k, v in self._pending_metrics.items():
                logs.setdefault(k, v)
            self._pending_metrics = {}
        return super().log(logs, start_time, **kwargs)

    # ------------------------------------------------------------------ #
    # Core GRPO math
    # ------------------------------------------------------------------ #
    def _build_reference_model(self, model):
        ref = copy.deepcopy(model)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        return ref

    @torch.no_grad()
    def _sample_completions(self, model, flat_prompts):
        tok = self.processing_class
        was_training = model.training
        model.eval()
        prompt_ids_list, completion_ids_list = [], []
        temperature = float(getattr(self.args, "temperature", 1.0))
        do_sample = temperature > 0.0
        for p in flat_prompts:
            enc = tok(p, truncation=True, max_length=int(self.args.max_prompt_length), return_tensors="pt")
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            gen = dict(
                max_new_tokens=int(self.args.max_completion_length),
                do_sample=do_sample,
                pad_token_id=self._pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            if do_sample:
                gen["temperature"] = temperature
                gen["top_p"] = float(getattr(self.args, "top_p", 1.0))
                top_k = getattr(self.args, "top_k", 0)
                if top_k and top_k > 0:
                    gen["top_k"] = int(top_k)
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen)
            completion = out[0, input_ids.shape[1]:]
            prompt_ids_list.append(input_ids[0].cpu())
            completion_ids_list.append(completion.cpu())
        if was_training:
            model.train()
        return prompt_ids_list, completion_ids_list

    def _build_training_batch(self, prompt_ids_list, completion_ids_list):
        pad = self._pad_token_id
        seqs, masks = [], []
        for pids, cids in zip(prompt_ids_list, completion_ids_list):
            seqs.append(torch.cat([pids, cids]))
            masks.append(
                torch.cat(
                    [
                        torch.zeros(len(pids), dtype=torch.long),
                        torch.ones(len(cids), dtype=torch.long),
                    ]
                )
            )
        max_len = max(s.shape[0] for s in seqs)
        B = len(seqs)
        input_ids = torch.full((B, max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        completion_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, (seq, mask) in enumerate(zip(seqs, masks)):
            L = seq.shape[0]
            input_ids[i, :L] = seq
            attention_mask[i, :L] = 1
            completion_mask[i, :L] = mask
        return input_ids, attention_mask, completion_mask

    def _per_token_logps(self, model, input_ids, attention_mask, completion_mask, track_grad=False):
        ctx = nullcontext() if track_grad else torch.no_grad()
        with ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = completion_mask[:, 1:].float()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        return gathered * shift_mask

    def _ref_logps(self, model, input_ids, attention_mask, completion_mask):
        if self.ref_model is None:
            if self._is_peft and hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    return self._per_token_logps(model, input_ids, attention_mask, completion_mask, track_grad=False)
            return None
        return self._per_token_logps(self.ref_model, input_ids, attention_mask, completion_mask, track_grad=False)

    def _per_token_entropy(self, model, input_ids, attention_mask):
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            lp = F.log_softmax(out.logits.float()[:, :-1, :], dim=-1)
            return -(lp.exp() * lp).sum(-1)

    def _build_reward_batch(self, flat_prompts, completion_ids_list):
        batch = []
        for p_text, cids in zip(flat_prompts, completion_ids_list):
            batch.append(
                {
                    "prompt": p_text,
                    "completion": self.processing_class.decode(cids, skip_special_tokens=False),
                    "completion_ids": cids.tolist(),
                }
            )
        return batch

    def _compute_rewards(self, reward_batch):
        if isinstance(self.reward_funcs, (list, tuple)):
            all_scores = [rf(reward_batch) for rf in self.reward_funcs]
            if all_scores and isinstance(all_scores[0][0], (int, float)):
                return [sum(vals) for vals in zip(*all_scores)]
            return all_scores[0]
        return self.reward_funcs(reward_batch)

    def _group_advantages(self, rewards):
        G = int(self.args.num_generations)
        P = len(rewards) // G
        r = torch.tensor(rewards, dtype=torch.float32).view(P, G)
        mean = r.mean(dim=1, keepdim=True)
        std = r.std(dim=1, keepdim=True, unbiased=False)
        adv = (r - mean) / (std + 1e-4)
        self._last_reward_std = std.mean()
        return adv.view(-1)

    def _grpo_forward_and_metrics(self, model, prompts):
        args = self.args
        G = int(args.num_generations)
        flat_prompts = [p for p in prompts for _ in range(G)]
        device = model.device

        prompt_ids_list, completion_ids_list = self._sample_completions(model, flat_prompts)
        input_ids, attention_mask, completion_mask = self._build_training_batch(prompt_ids_list, completion_ids_list)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        completion_mask = completion_mask.to(device)

        shift_mask = completion_mask[:, 1:].float()
        denom = shift_mask.sum().clamp(min=1.0)

        current_logps = self._per_token_logps(model, input_ids, attention_mask, completion_mask, track_grad=True)
        old_logps = self._per_token_logps(model, input_ids, attention_mask, completion_mask, track_grad=False)
        ref_logps = self._ref_logps(model, input_ids, attention_mask, completion_mask)
        kl_per_token = (current_logps - ref_logps) if ref_logps is not None else torch.zeros_like(current_logps)

        rewards = self._compute_rewards(self._build_reward_batch(flat_prompts, completion_ids_list))
        advantages = self._group_advantages(rewards).to(device)
        reward_mean = torch.tensor(rewards, dtype=torch.float32).mean()

        adv_tok = advantages.unsqueeze(1) * shift_mask
        eps_clip = float(getattr(args, "epsilon", 0.2))
        coef_1 = torch.exp(current_logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1.0 - eps_clip, 1.0 + eps_clip)
        ptl1 = coef_1 * adv_tok
        ptl2 = coef_2 * adv_tok
        ptl = -torch.min(ptl1, torch.where(adv_tok < 0, ptl2, ptl1))
        pg_loss = (ptl * shift_mask).sum() / denom
        clip_ratio = ((torch.abs(coef_1 - 1.0) > eps_clip).float() * shift_mask).sum() / denom

        ent = self._per_token_entropy(model, input_ids, attention_mask)
        entropy = (ent * shift_mask).sum() / denom

        kl_loss = (kl_per_token * shift_mask).sum() / denom
        beta = float(getattr(args, "beta", 0.0))
        loss = pg_loss + beta * kl_loss
        completion_length = completion_mask.sum(dim=1).float().mean()

        metrics = {
            "loss": loss.detach(),
            "reward": reward_mean.detach(),
            "reward_std": self._last_reward_std.detach(),
            "kl": kl_loss.detach(),
            "entropy": entropy.detach(),
            "clip_ratio": clip_ratio.detach(),
            "completion_length": completion_length.detach(),
        }
        self._pending_metrics = {k: float(v.item()) for k, v in metrics.items()}
        return {"loss": loss, "metrics": metrics}