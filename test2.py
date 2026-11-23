import math, torch, logging
from types import SimpleNamespace

logger = logging.getLogger("t")
def _first_attr(args, names, default):
    for n in names:
        if hasattr(args, n):
            return getattr(args, n)
    return default

def _group_advantages(self, rewards):
    G = int(self.args.num_generations)
    if len(rewards) % G != 0:
        raise ValueError(f"Number of rewards ({len(rewards)}) is not divisible by num_generations ({G}).")
    P = len(rewards) // G
    r = torch.tensor(rewards, dtype=torch.float32).view(P, G)
    mean = r.mean(1, keepdim=True)
    std = r.std(1, keepdim=True, unbiased=False)
    self._last_reward_std = std.mean()
    return ((r - mean) / std.clamp(min=1e-8)).view(-1)

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
    # [Fix] slice entropy to [B, T-1] so it aligns with shift_mask:
    # entropy[t] is from the logit at position t, which predicts token t+1
    entropy = (self._token_entropy(logits.detach())[:, :-1] * shift_mask).sum() / denom
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

# Minimal helpers (match your real ones' shapes: per-token, [B, T-1])
def _token_logps(self, logits, input_ids):
    logp = logits.log_softmax(-1)
    return logp[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

def _token_entropy(self, logits):
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)

class GRPOTrainer:
    _group_advantages = _group_advantages
    _grpo_forward = _grpo_forward
    _token_logps = _token_logps
    _token_entropy = _token_entropy

# ---- Test harness (no model.generate()) ----
torch.manual_seed(0)
G, B, T, V = 2, 4, 8, 16
REWARDS = [0.0, 1.0, 0.5, 0.5]  # 2nd group constant -> exercises std.clamp

class DummyModel:
    device = torch.device("cpu")
    def __call__(self, input_ids, attention_mask, use_cache=False):
        self.last_logits = torch.randn(B, T, V, requires_grad=True)
        return SimpleNamespace(logits=self.last_logits)

t = object.__new__(GRPOTrainer)
t.args = SimpleNamespace(num_generations=G, beta=0.0, epsilon=0.2)
t.ref_model = None
t._is_peft = False

gen = torch.Generator().manual_seed(1)
input_ids = torch.randint(0, V, (B, T), generator=gen)
cmask = torch.zeros(B, T); cmask[:, 4:] = 1  # completions at positions 4..7
t._sample_completions = lambda m, p: (list(range(B)), torch.arange(B) * G, torch.ones(B) * 3)
t._build_training_batch = lambda a, b: (input_ids, torch.ones_like(input_ids), cmask)
t._compute_rewards = lambda pr, ci: list(REWARDS)

m = DummyModel()
out = t._grpo_forward(m, list(range(B)))

adv = t._group_advantages(REWARDS)
assert math.isclose(adv.mean().item(), 0.0, abs_tol=1e-5), f"advantages not zero-mean: {adv.mean().item()}"

out["loss"].backward()
g = m.last_logits.grad
assert g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0, "bad gradient"
print(f"OK  adv mean={adv.mean().item():.2e}  loss={out['loss'].item():.4f}  "
      f"|grad| sum={g.abs().sum().item():.4f}  finite={bool(torch.isfinite(g).all())}")

# ---- Second check: beta=0.1 + ref_model -> KL loss & grads must be finite & nonzero ----
t.args.beta = 0.1
t.ref_model = DummyModel()

m2 = DummyModel()
out2 = t._grpo_forward(m2, list(range(B)))

kl_val = out2["metrics"]["kl"]
assert torch.isfinite(kl_val) and kl_val.item() > 0.0, f"kl not finite or zero: {kl_val.item()}"
assert torch.isfinite(out2["loss"]), f"loss not finite: {out2['loss'].item()}"

out2["loss"].backward()
g2 = m2.last_logits.grad
assert g2 is not None and torch.isfinite(g2).all() and g2.abs().sum().item() > 0, "bad gradient (kl run)"
print(f"OK2 kl={kl_val.item():.4f}  loss={out2['loss'].item():.4f}  "
      f"|grad| sum={g2.abs().sum().item():.4f}  finite={bool(torch.isfinite(g2).all())}")