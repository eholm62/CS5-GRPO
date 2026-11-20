import torch
import torch.nn as nn
import logging
from typing import List, Dict, Optional, Tuple

# Set up logging for better error visibility
logger = logging.getLogger(__name__)

class GRPOSTrainer:
    """
    Improved Implementation of Group Relative Policy Optimization (GRPO).
    Handles non-divisible batch sizes and provides robust normalization.
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
        self.eps_stable = eps_stable  # Centralized epsilon for stability
        self.history = {"reward": [], "kl": [], "loss": []}

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes advantages using group-relative normalization.
        Gracefully handles batch sizes that are not multiples of group_size by 
        truncating the remainder.
        """
        if not isinstance(rewards, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for rewards, got {type(rewards)}")
        
        if rewards.dim() != 1:
            # Ensure rewards is 1D even if passed as [N, 1]
            rewards = rewards.view(-1)

        total_samples = rewards.size(0)
        
        # Graceful handling of non-divisible batches: 
        # Instead of crashing, we use the largest possible multiple of group_size.
        num_groups = total_samples // self.group_size
        if num_groups == 0:
            logger.warning("Batch size is smaller than group_size. Returning zero advantages.")
            return torch.zeros_like(rewards)

        # Truncate rewards to fit perfect groups
        effective_size = num_groups * self.group_size
        usable_rewards = rewards[:effective_size]
        
        # Reshape for vectorized group operations
        grouped_rewards = usable_rewards.view(num_groups, self.group_size)

        # Compute group statistics
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        group_stds = grouped_rewards.std(dim=1, keepdim=True)

        # Normalize within each group
        # We add eps_stable to prevent division by zero if a group has identical rewards
        normalized_within_group = (grouped_rewards - group_means) / (group_stds + self.eps_stable)

        # Flatten back to match original batch (padding the remainder with 0 advantage)
        advantages = torch.zeros_like(rewards)
        advantages[:effective_size] = normalized_within_group.view(-1)

        # Optional: Global normalization for training stability 
        # (Only applied to the valid portion to avoid bias from the zero-padding)
        if effective_size > 0:
            valid_adv = advantages[:effective_size]
            global_mean = valid_adv.mean()
            global_std = valid_adv.std()
            advantages[:effective_size] = (valid_adv - global_mean) / (global_std + self.eps_stable)

        return advantages

    def compute_loss(
        self,
        response_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the GRPO loss with input validation.
        """
        # Basic validation to catch shape mismatches early
        if not (response_ids.size(0) == rewards.size(0) == old_log_probs.size(0)):
            raise ValueError(
                f"Dimension mismatch: response_ids({response_ids.size(0)}), "
                f"rewards({rewards.size(0)}), old_log_probs({old_log_probs.size(0)})"
            )

        # Logic for loss calculation would follow here...
        # (Placeholder for brevity)
        loss = torch.tensor(0.0, requires_grad=True)
        metrics = {"loss": 0.0}
        return loss, metrics
