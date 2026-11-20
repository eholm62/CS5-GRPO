import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

class GRPOSTrainer:
    """
    Implementation of Group Relative Policy Optimization (GRPO) following the original specification.
    """
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        kl_coeff: float = 0.01,
        group_size: int = 8,
        eps_clip: float = 0.2
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        self.group_size = group_size
        self.eps_clip = eps_clip
        self.history = {"reward": [], "kl": [], "loss": []}

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes advantages using group-relative normalization as per GRPO specification.
        Args:
            rewards: Tensor of shape (batch_size,) containing rewards for each sample
        Returns:
            Normalized advantages tensor of same shape as input
        """
        # Group the rewards
        num_groups = len(rewards) // self.group_size
        if len(rewards) % self.group_size != 0:
            raise ValueError("Number of samples must be divisible by group size")

        grouped_rewards = rewards.view(num_groups, self.group_size)

        # Compute group statistics
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        group_stds = grouped_rewards.std(dim=1, keepdim=True) + 1e-8

        # Normalize within each group
        normalized_within_group = (grouped_rewards - group_means) / group_stds

        # Compute global statistics for final normalization
        global_mean = normalized_within_group.mean()
        global_std = normalized_within_group.std() + 1e-8

        # Final normalization to have mean 0 and std 1 across all groups
        advantages = (normalized_within_group - global_mean) / global_std

        return advantages.view(-1)

    def compute_loss(
        self,
        response_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the GRPO loss following the original specification.
        """
        # 1. Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards)

        # 2. Get current log probabilities from policy and reference models
        with torch.no_grad():
            ref_outputs = self.ref_model(response_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits

        policy_outputs = self.policy(response_ids, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits

        # 3. Compute log probabilities (shifted for language modeling)
        shift_mask = torch.zeros_like(attention_mask)
        shift_mask[:, 1:] = attention_mask[:, :-1]
        per_token_ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        per_token_policy_log_probs = F.log_softmax(policy_logits, dim=-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        # Apply shift mask
        ref_log_probs = (per_token_ref_log_probs * shift_mask).sum(dim=1)
        policy_log_probs = (per_token_policy_log_probs * shift_mask).sum(dim=1)

        # 4. Compute KL divergence between old and new policies
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            F.softmax(ref_logits.detach(), dim=-1),
            reduction='none'
        ).sum(dim=-1)

        # 5. Compute GRPO loss components
        ratio = torch.exp(policy_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        kl_loss = (kl_div * self.kl_coeff).mean()

        total_loss = policy_loss + kl_loss

        # Log metrics
        metrics = {
            "reward_mean": rewards.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "kl_mean": kl_div.mean().item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item()
        }

        return total_loss, metrics
