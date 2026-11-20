import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np

class GRPOSTrainer:
    """
    Implementation of Group Relative Policy Optimization (GRPO).
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
        Computes the advantage using group-relative normalization.
        """
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        return (rewards - mean) / std

    def get_log_probs(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper to get log probabilities. 
        Corrected to handle the shift: logits at index 't' predict token at 't+1'.
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Shift logits and labels: 
        # We want the log_prob of the token that actually appeared.
        # Logits at position t are used to predict token at position t+1.
        # Therefore, we shift logits left and input_ids right.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather the log probabilities of the actual tokens
        per_token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding and return
        return per_token_log_probs * shift_mask

    def compute_loss(
        self, 
        response_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor  # FIX: Now passed from sampling phase
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the GRPO loss.
        """
        # 1. Advantage Calculation (G,)
        advantages = self.compute_group_advantages(rewards)

        # 2. Get current Log Probs from Policy and Reference
        # policy_log_probs: (G, seq_len-1)
        policy_log_probs = self.get_log_probs(self.policy, response_ids, attention_mask)
        ref_log_probs = self.get_log_probs(self.ref_model, response_ids, attention_mask)

        # 3. Policy Ratio
        # ratio is calculated per token, then we aggregate per sequence.
        # FIX: ratio calculation is now meaningful because old_log_probs are from sampling.
        ratio = torch.exp(policy_log_probs - old_log_probs)

        # 4. KL Divergence (Per-token)
        kl_div = (policy_log_probs - ref_log_probs)

        # 5. GRPO Loss Core
        # Sum log_probs over sequence to get per-completion values
        seq_log_probs = policy_log_probs.sum(dim=-1)
        seq_old_log_probs = old_log_probs.sum(dim=-1)
        seq_kl = kl_div.sum(dim=-1)

        # Re-calculate sequence-level ratio for the surrogate objective
        # Note: In GRPO/PPO, the ratio is typically applied to the sum of log_probs 
        # or we use the per-token ratio. For stability, we use per-token ratio for clipping.
        
        # Clipped Surrogate Objective (Per-token)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages.unsqueeze(-1)
        
        # We take the mean of the minimum surr across tokens, then mean across group
        # (Or sum across tokens, then mean across group)
        policy_loss = -torch.min(surr1, surr2).mean()

        # 6. KL Regularization term
        kl_loss = self.kl_coeff * seq_kl.mean()

        total_loss = policy_loss + kl_loss

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": seq_kl.mean().item()
        }

        return total_loss, metrics

# --- MOCK CLASSES AND TEST EXECUTION ---

class MockModel(nn.Module):
    def __init__(self, vocab_size=100, seq_len=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        logits = torch.randn(batch_size, self.seq_len, self.vocab_size, device=input_ids.device)
        class Output:
            def __init__(self, logits): self.logits = logits
        return Output(logits)

if __name__ == "__main__":
    G = 4 
    SEQ_LEN = 10
    VOCAB_SIZE = 50

    policy = MockModel(VOCAB_SIZE, SEQ_LEN)
    ref = MockModel(VOCAB_SIZE, SEQ_LEN)
    trainer = GRPOSTrainer(policy, ref, group_size=G)

    # SIMULATED SAMPLING STEP (The missing piece)
    sample_response_ids = torch.randint(0, VOCAB_SIZE, (G, SEQ_LEN))
    sample_mask = torch.ones((G, SEQ_LEN))
    sample_rewards = torch.tensor([1.0, 2.5, 0.5, 4.0])

    # To correctly implement PPO/GRPO, we MUST capture log_probs during sampling
    # We simulate this by calling get_log_probs on the current policy BEFORE optimization
    with torch.no_grad():
        sampled_log_probs = trainer.get_log_probs(policy, sample_response_ids, sample_mask)

    print("Starting Fixed GRPO Training Step...")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    optimizer.zero_grad()
    loss, metrics = trainer.compute_loss(
        response_ids=sample_response_ids,
        attention_mask=sample_mask,
        rewards=sample_rewards,
        old_log_probs=sampled_log_probs # Pass the correct "old" values
    )
    loss.backward()
    optimizer.step()

    print(f"Success! Metrics: {metrics}")
