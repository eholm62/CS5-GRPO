import torch
import torch.nn as nn
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class GRPOSTrainer:
    """
    Refined Implementation of Group Relative Policy Optimization (GRPO).
    Strictly adheres to group-relative advantage normalization.
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
        """
        Computes advantages using strict group-relative normalization.
        Handles non-divisible batch sizes by treating the remainder as a final, 
        smaller group to ensure every sample receives a valid relative advantage.
        """
        if not isinstance(rewards, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for rewards, got {type(rewards)}")
        if rewards.dim() != 1:
            raise ValueError(f"Expected 1D tensor for rewards, got {rewards.dim()}D")

        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        
        # Calculate how many full groups we have and the remainder
        num_full_groups = batch_size // self.group_size
        remainder = batch_size % self.group_size

        # 1. Process full groups using vectorized operations for efficiency
        if num_full_groups > 0:
            usable_rewards = rewards[:num_full_groups * self.group_size]
            grouped_rewards = usable_rewards.view(num_full_groups, self.group_size)
            
            group_means = grouped_rewards.mean(dim=1, keepdim=True)
            group_stds = grouped_rewards.std(dim=1, keepdim=True)
            
            # Normalize within groups
            normalized_groups = (grouped_rewards - group_means) / (group_stds + self.eps_stable)
            advantages[:num_full_groups * self.group_size] = normalized_groups.view(-1)

        # 2. Process remainder as a single smaller group (if it exists)
        # This ensures the remainder samples are not biased with 0-advantage
        if remainder > 0:
            start_idx = num_full_groups * self.group_size
            end_idx = batch_size
            remainder_rewards = rewards[start_idx:end_idx]
            
            rem_mean = remainder_rewards.mean()
            rem_std = remainder_rewards.std()
            
            # If remainder group has 0 variance, std is 0; eps_stable handles it
            advantages[start_idx:end_idx] = (remainder_rewards - rem_mean) / (rem_std + self.eps_stable)

        return advantages

    def compute_loss(
        self,
        response_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the GRPO loss. (Placeholder for logic involving PPO-style clipping 
        and KL divergence penalty as per GRPO spec).
        """
        # Implementation would proceed here with advantages from compute_group_advantages
        pass
