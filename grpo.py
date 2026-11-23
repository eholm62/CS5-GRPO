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
from transformers import Trainer, GenerationConfig
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
        self._last_reward_std = torch.tensor(0.0)

        if args is not None:
            args.remove_unused_columns = False

        # PEFT
        self._is_peft = peft_config is not None
        if self._is_peft:
            if not is_peft_available():
                raise ImportError("peft must be installed for peft_config.")
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)

        # Processing class
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

        # Warn early if PEFT + beta > 0 but no disable_adapter available
        if beta > 0.0 and self._is_peft:
            if not hasattr(model, "disable_adapter"):
                logger.warning(
                    "GRPOTrainer: beta=%.4f is set with PEFT but the model has "
                    "no 'disable_adapter' method. The KL penalty will be "
                    "silently zeroed out. Consider using a PEFT model that "
                    "supports disable_adapter or setting beta=0.",
                    beta,
                )

        if optimizers is None:
            optimizers = (None, None)

        trainer_kwargs: dict = dict(
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

    # ------------------------------------------------------------------
    # Processing class inference
    # ------------------------------------------------------------------

    def _infer_processing_class(self, model):
        model_name = None
        if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        elif hasattr(model, "name_or_path"):
            model_name = model.name_or_path
        if model_name is None:
            raise ValueError("Cannot infer a processing_class from the model.")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_collate(examples):
        return {"prompts": [ex["prompt"] for ex in examples]}

    def _prompts_per_batch(self):
        # [Fix A] TRL semantics: per_device_train_batch_size counts PROMPTS.
        # G generations per prompt are produced internally during the forward.
        return max(int(self.args.per_device_train_batch_size), 1)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        seed = _first_attr(self.args, ("data_seed", "seed"), 42)
        shuffle = getattr(self.args, "shuffle", True)
        return DataLoader(
            self.train_dataset,
            batch_size=self._prompts_per_batch(),
            shuffle=shuffle,
            collate_fn=self._prompt_collate,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            generator=torch.Generator().manual_seed(int(seed)),
        )

    # ------------------------------------------------------------------
    # Loss / metrics hooks
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        num_items_in_batch is accepted for interface compatibility with
        newer HF Trainer versions. Our loss is already normalised
        per-token (mean over valid completion tokens), so no additional
        scaling is needed.
        """
        result = self._grpo_forward(model, inputs["prompts"])
        if return_outputs:
            return result["loss"], result
        return result["loss"]

    def compute_loss_and_metrics(self, model, inputs, num_items_in_batch=None):
        """Return (loss, metrics_dict) where metrics_dict is a flat dict[str, float]."""
        loss, result = self.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        metrics = {
            k: (float(v.item()) if hasattr(v, "item") else float(v))
            for k, v in result["metrics"].items()
            if k != "loss"  # Trainer logs its own loss
        }
        return loss, metrics

    def log(self, logs, start_time=None, **kwargs):
        """
        Inject GRPO-specific metrics only; do NOT overwrite "loss" —
        the Trainer's logged loss is authoritative.
        """
        if self._pending_metrics:
            for k, v in self._pending_metrics.items():
                if k == "loss":
                    continue
                logs.setdefault(k, v)
            self._pending_metrics = {}
        return super().log(logs, start_time, **kwargs)

    # ------------------------------------------------------------------
    # Length helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_completions(self, model, prompts):
        """
        [Fix B] One generate() call per BATCH, with rows in
        [p0 x G, p1 x G, ...] order — matches TRL's generation batching
        so the sampling RNG stream (and therefore completions under a
        fixed seed) is identical to TRL's.

        [Fix C] An explicit GenerationConfig is passed, which REPLACES
        the model's hub generation_config (e.g. opt-125m ships
        no_repeat_ngram_size=3, max_length=20, pad_token_id=1). This is
        what TRL does; leaking the hub config was a major source of
        divergence (and a suspected trigger for multinomial failures).

        [Fix D] The previous try/except RuntimeError -> greedy fallback
        is REMOVED. It masked non-finite-logit failures and silently
        produced degenerate completions (reward=0.0 in the unit test).
        A real generation failure now raises, at the exact failing step,
        just like TRL would.

        [Fix F] Prompts in one batch have different token lengths, and
        torch.cat(dim=0) requires identical sizes in all other dims
        (crashed with "Expected size 7 but got size 8"). Prompts are
        therefore LEFT-PADDED with pad_token_id to the batch max length
        (with a matching attention mask of 0s over the pad region)
        before the single generate() call — the same thing TRL's
        prompt collator does. Left-padding (rather than right-padding)
        is required for causal-LM generation so padded positions are not
        attended as if they were real context tokens. After padding,
        every batched row has the same prompt length L_pad, so the
        completion for row r is simply out[r, L_pad:].

        Completions are trimmed at the first EOS token.
        """
        tok = self.processing_class
        G = int(self.args.num_generations)
        temperature = float(getattr(self.args, "temperature", 1.0) or 1.0)
        top_p = float(getattr(self.args, "top_p", 1.0) or 1.0)
        top_k = int(getattr(self.args, "top_k", 0) or 0)
        do_sample = temperature > 0.0

        eos_id = tok.eos_token_id if tok.eos_token_id is not None else self._pad_token_id

        # Encode all prompts in the batch (original, unpadded).
        encs, prompt_lens = [], []
        for p in prompts:
            enc = tok(
                p,
                truncation=True,
                max_length=self._max_prompt_length(),
                return_tensors="pt",
            )
            encs.append(enc)
            prompt_lens.append(enc["input_ids"].shape[1])

        # [Fix F] Left-pad prompts to the batch's max prompt length so
        # they can form a rectangular tensor for a single generate call.
        L_pad = max(prompt_lens)
        padded_ids, padded_am = [], []
        for enc, pl in zip(encs, prompt_lens):
            ids = enc["input_ids"][0]
            pad_len = L_pad - pl
            if pad_len > 0:
                ids = torch.cat(
                    [
                        torch.full((1, pad_len), self._pad_token_id, dtype=torch.long),
                        ids,
                    ],
                    dim=1,
                )
                am = torch.cat(
                    [
                        torch.zeros((1, pad_len), dtype=torch.long),
                        enc["attention_mask"][0],
                    ],
                    dim=1,
                )
            else:
                am = enc["attention_mask"][0]
            padded_ids.append(ids)
            padded_am.append(am)

        base_ids = torch.stack(padded_ids, dim=0)     # [P, L_pad]
        base_am = torch.stack(padded_am, dim=0)       # [P, L_pad]

        # [Fix B] Batch all G copies: rows in [p0 x G, p1 x G, ...] order.
        input_ids = base_ids.repeat_interleave(G, dim=0).to(model.device)
        attention_mask = base_am.repeat_interleave(G, dim=0).to(model.device)

        # [Fix C] Explicit GenerationConfig replaces the hub config.
        generation_config = GenerationConfig(
            max_new_tokens=self._max_completion_length(),
            do_sample=do_sample,
            pad_token_id=self._pad_token_id,
            eos_token_id=eos_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        was_training = model.training
        model.eval()
        try:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
        finally:
            if was_training:
                model.train()

        # Split the output back into per-sample completions. Every row
        # shares the padded prompt length L_pad, so the completion for
        # row r is out[r, L_pad:], trimmed at the first EOS token.
        flat_prompts, pids_list, cids_list = [], [], []
        row = 0
        for i, p in enumerate(prompts):
            for _ in range(G):
                flat_prompts.append(p)
                # Keep the ORIGINAL (unpadded) prompt ids for the training batch.
                pids_list.append(encs[i]["input_ids"][0].cpu())

                completion = out[row, L_pad:]
                if eos_id is not None:
                    eos_positions = (completion == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        completion = completion[: eos_positions[0].item() + 1]
                cids_list.append(completion.cpu())
                row += 1

        return flat_prompts, pids_list, cids_list

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _build_training_batch(self, pids_list, cids_list):
        """
        Completions are EOS-trimmed (no post-EOS padding), so the mask is
        naturally correct: all completion positions are real tokens, and
        any inter-sample padding added here is masked as 0 in both
        attn and cmask.
        """
        pad = self._pad_token_id
        seqs, cms = [], []
        for p, c in zip(pids_list, cids_list):
            seqs.append(torch.cat([p, c]))
            cms.append(
                torch.cat(
                    [
                        torch.zeros(len(p), dtype=torch.long),
                        torch.ones(len(c), dtype=torch.long),
                    ]
                )
            )
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

    # ------------------------------------------------------------------
    # Token-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_logps(logits, input_ids):
        """log-prob of input_ids[1:] given logits[:-1]. Shape: [B, L-1]."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        return torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _token_entropy(logits):
        """Entropy of the distribution at each position. Shape: [B, L-1]."""
        shift_logits = logits[:, :-1, :].contiguous()
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        return -(probs * log_probs).sum(-1)

    # ------------------------------------------------------------------
    # Rewards and advantages
    # ------------------------------------------------------------------

    def _compute_rewards(self, flat_prompts, cids_list):
        """Use self.processing_class (the primary tokenizer) for decoding."""
        proc = self.processing_class
        completions = [proc.decode(c, skip_special_tokens=True) for c in cids_list]
        cids = [c.tolist() for c in cids_list]
        all_scores = [
            rf(prompts=flat_prompts, completions=completions, completion_ids=cids)
            for rf in self.reward_funcs
        ]
        if len(all_scores) == 1:
            return [float(s) for s in all_scores[0]]
        return [float(sum(v)) for v in zip(*all_scores)]

    def _group_advantages(self, rewards):
        G = int(self.args.num_generations)
        if len(rewards) % G != 0:
            raise ValueError(
                f"Number of rewards ({len(rewards)}) is not divisible by "
                f"num_generations ({G}). Cannot form complete groups."
            )
        P = len(rewards) // G
        r = torch.tensor(rewards, dtype=torch.float32).view(P, G)
        mean = r.mean(1, keepdim=True)
        std = r.std(1, keepdim=True, unbiased=False)
        self._last_reward_std = std.mean()
        # clamp instead of additive epsilon to match TRL
        return ((r - mean) / std.clamp(min=1e-8)).view(-1)

    # ------------------------------------------------------------------
    # Core GRPO forward
    # ------------------------------------------------------------------

    def _grpo_forward(self, model, prompts):
        """
        Key design points:
        - Single forward pass for old AND current log-probs. For
          single-step on-policy GRPO, old_policy == current_policy,
          so old_logps = cur_logps.detach() and ratio = 1 exactly.
          This avoids any dropout/ordering mismatch between two
          separate forward passes and saves one pass.
        - [Fix E] k3 KL estimator: exp(ref - cur) - (ref - cur) - 1,
          matching TRL's GRPOTrainer (GRPO paper, Eq. 6).
        """
        args = self.args
        G = int(getattr(args, "num_generations", 1))
        device = model.device

        # --- Sample completions ---
        flat_prompts, pids, cids = self._sample_completions(model, prompts)
        input_ids, attn, cmask = self._build_training_batch(pids, cids)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        cmask = cmask.to(device)

        # Mask for completion-token log-probs (shifted by 1 for alignment):
        # cur_logps[t] = log P(input_ids[t+1] | input_ids[:t+1])
        # shift_mask[t] = cmask[t+1] => 1 when the token BEING PREDICTED
        # is a completion token.
        shift_mask = cmask[:, 1:].float()
        denom = shift_mask.sum().clamp(min=1.0)

        # --- Single forward pass: derive both old and current log-probs ---
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits.float()
        cur_logps = self._token_logps(logits, input_ids)
        old_logps = cur_logps.detach()  # Same values, no gradient

        # --- Reference log-probs ---
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
            if float(getattr(args, "beta", 0.0) or 0.0) > 0.0:
                logger.warning(
                    "GRPOTrainer: ref_logps unavailable (no ref_model and no "
                    "disable_adapter). KL penalty will be zero."
                )

        # --- Ratio and KL ---
        ratio = torch.exp(cur_logps - old_logps)
        # [Fix E] k3 KL estimator, matching TRL's GRPOTrainer
        if ref_logps is not None:
            kl_tok = torch.exp(ref_logps - cur_logps) - (ref_logps - cur_logps) - 1.0
        else:
            kl_tok = torch.zeros_like(cur_logps)

        # --- Rewards and advantages ---
        rewards = self._compute_rewards(flat_prompts, cids)
        adv = self._group_advantages(rewards).to(device)
        adv_tok = adv.unsqueeze(1)  # [B, 1] broadcast over tokens

        # --- PPO surrogate loss ---
        eps = float(_first_attr(args, ("epsilon", "epsilon_low"), 0.2) or 0.2)
        surr1 = ratio * adv_tok
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_tok
        pg_loss = -torch.min(surr1, surr2).mul(shift_mask).sum() / denom

        # --- KL penalty ---
        kl_loss = (kl_tok * shift_mask).sum() / denom

        # --- Total loss ---
        beta = float(getattr(args, "beta", 0.0) or 0.0)
        loss = pg_loss + beta * kl_loss

        # --- Metrics (flat dict of detached tensors) ---
        clip_frac = ((ratio - 1.0).abs() > eps).float().mul(shift_mask).sum() / denom
        entropy = (self._token_entropy(logits.detach()) * shift_mask).sum() / denom
        comp_len = cmask.sum(1).float().mean()

        metrics = {
            "loss": loss.detach(),
            "reward": torch.tensor(rewards, dtype=torch.float32).mean().detach(),
            "reward_std": self._last_reward_std.detach(),
            "kl": kl_loss.detach(),
            "entropy": entropy.detach(),
            "clip_ratio": clip_frac.detach(),
            "completion_length": comp_len.detach(),
        }

        # Store for log() — Trainer's own "loss" takes precedence
        self._pending_metrics = {
            k: (float(v.item()) if hasattr(v, "item") else float(v))
            for k, v in metrics.items()
        }

        return {"loss": loss, "metrics": metrics}