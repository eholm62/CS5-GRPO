import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

class GRPOSTrainer:
    """
    Production-aligned Implementation of Group Relative Policy Optimization (GRPO).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        kl_coeff: float = 0.01,
        group_size: int = 8,
        eps_clip: float = 0.2,
        eps_stable: float = 1e-8
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        self.group_size = group_size
        self.eps_clip = eps_clip
        self.eps_stable = eps_stable
        self.history = {"reward": [], "kl": [], "loss": []}

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Computes group-relative advantages: (r - mean) / std."""
        mean = rewards.mean()
        std = rewards.std() + self.eps_stable
        return (rewards - mean) / std

    def get_log_probs(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extracts log probabilities for the provided token IDs."""
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Standard causal LM shift: align logits[t] with tokens[t+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return per_token_log_probs * shift_mask

    def compute_loss(
        self, 
        response_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor  # CRITICAL: Pass log_probs from the sampling step
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the GRPO loss using true Importance Sampling.
        """
        # 1. Advantages
        advantages = self.compute_group_advantages(rewards) # (G,)

        # 2. Get current policy and reference log probs
        current_log_probs = self.get_log_probs(self.policy, response_ids, attention_mask)
        ref_log_probs = self.get_log_probs(self.ref_model, response_ids, attention_mask)

        # 3. Importance Sampling Ratio
        # old_log_probs must be the sum of log_probs per sequence from the sampling phase
        seq_current_log_probs = current_log_probs.sum(dim=-1)
        seq_old_log_probs = old_log_probs.sum(dim=-1)
        
        ratio = torch.exp(seq_current_log_probs - seq_old_log_probs)

        # 4. Clipped Surrogate Objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. KL Divergence Regularization (Per-token)
        # We use the sequence-level KL to ensure stability
        kl_div = (current_log_probs - ref_log_probs).mean()
        kl_loss = self.kl_coeff * kl_div

        total_loss = policy_loss + kl_loss

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": rewards.mean().item()
        }

        return total_loss, metrics

    def log_step(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.history[k].append(v)
