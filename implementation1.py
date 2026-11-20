import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np

class GRPOSTrainer:
    """
    Implementation of Group Relative Policy Optimization (GRPO).
    
    GRPO eliminates the need for a Critic (Value) model by estimating advantages 
    relative to a group of sampled outputs for the same prompt.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        kl_coeff: float = 0.01,
        group_size: int = 8,
        eps_clip: float = 0.2
    ):
        """
        Initialize the GRPO Trainer.

        Args:
            policy_model: The active policy being optimized.
            ref_model: The frozen reference model used for KL regularization.
            kl_coeff: Coefficient for the KL divergence penalty.
            group_size (G): Number of outputs sampled per prompt.
            eps_clip: PPO-style clipping parameter.
        """
        self.policy = policy_model
        self.ref_model = ref_model
        self.kl_coeff = kl_coeff
        self.group_size = group_size
        self.eps_clip = eps_clip
        
        # History for logging
        self.history = {"reward": [], "kl": [], "loss": []}

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes the advantage using group-relative normalization.
        Formula: A_i = (r_i - mean(r)) / std(r)
        
        Args:
            rewards: Tensor of shape (group_size,) containing scalar rewards.
        """
        mean = rewards.mean()
        std = rewards.std() + 1e-8  # Stability epsilon
        return (rewards - mean) / std

    def get_log_probs(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper to get log probabilities for the sequences.
        Returns tensor of shape (batch, sequence_length).
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift logits and labels for next-token prediction alignment
        # log_probs shape: (batch, seq_len)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # We gather the log_probs of the actual tokens provided in input_ids
        # This assumes input_ids contains the response tokens we care about
        per_token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return per_token_log_probs

    def compute_loss(
        self, 
        prompt_ids: torch.Tensor, 
        response_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the GRPO loss including clipping and KL regularization.

        Args:
            prompt_ids: The prompt tokens.
            response_ids: The sampled response tokens.
            attention_mask: Mask for the combined prompt + response.
            rewards: Reward scores for each member of the group.

        Returns:
            total_loss: The scalar loss tensor.
            metrics: Dictionary of computed metrics for logging.
        """
        # 1. Advantage Calculation
        advantages = self.compute_group_advantages(rewards) # (G,)

        # 2. Get Log Probs from Policy and Reference
        # policy_log_probs: (G, seq_len)
        policy_log_probs = self.get_log_probs(self.policy, response_ids, attention_mask)
        # ref_log_probs: (G, seq_len)
        ref_log_probs = self.get_log_probs(self.ref_model, response_ids, attention_mask)

        # 3. Policy Ratio (for clipping)
        # In a real implementation, we would track old_log_probs from the sampling step
        # Here we assume we are doing a single-step optimization or use current as old
        # To strictly follow PPO clipping, we'd need the log_probs from the sampling phase.
        # For this modular demo, we use current policy.
        old_log_probs = policy_log_probs.detach() 
        ratio = torch.exp(policy_log_probs - old_log_probs)

        # 4. KL Divergence (Per-token approximation)
        # KL(pi || ref) = exp(log_ref - log_pi) - (log_ref - log_pi) is not standard;
        # we use the direct KL: log(pi) - log(ref)
        kl_div = (policy_log_probs - ref_log_probs)

        # 5. GRPO Loss Core
        # We aggregate log_probs over the sequence length (summing) to get per-completion log_prob
        # Masking is applied to ensure only non-pad tokens contribute
        token_mask = attention_mask.float()
        seq_log_probs = (policy_log_probs * token_mask).sum(dim=-1)
        seq_old_log_probs = (old_log_probs * token_mask).sum(dim=-1)
        seq_kl = (kl_div * token_mask).sum(dim=-1)

        ratio = torch.exp(seq_log_probs - seq_old_log_probs)
        
        # Clipped Surrogate Objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 6. KL Regularization term
        kl_loss = self.kl_coeff * seq_kl.mean()

        total_loss = policy_loss + kl_loss

        # Metrics for logging
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": seq_kl.mean().item()
        }

        return total_loss, metrics

    def log_step(self, metrics: Dict[str, float]):
        """Logs metrics to internal history."""
        for k, v in metrics.items():
            self.history[k].append(v)

# --- MOCK CLASSES FOR DEMONSTRATION ---

class MockModel(nn.Module):
    """A minimal model to allow the code to run out-of-the-box."""
    def __init__(self, vocab_size=100, seq_len=10):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, seq_len, vocab_size))
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def forward(self, input_ids, attention_mask=None):
        # Return dummy logits shaped like (batch, seq, vocab)
        batch_size = input_ids.shape[0]
        logits = torch.randn(batch_size, self.seq_len, self.vocab_size, device=input_ids.device)
        
        # Mocking the transformer output object
        class Output:
            def __init__(self, logits): self.logits = logits
        return Output(logits)

# --- TEST EXECUTION ---

if __name__ == "__main__":
    # Hyperparameters
    G = 4  # Group size
    SEQ_LEN = 10
    VOCAB_SIZE = 50

    # Initialize Models
    policy = MockModel(VOCAB_SIZE, SEQ_LEN)
    ref = MockModel(VOCAB_SIZE, SEQ_LEN)
    trainer = GRPOSTrainer(policy, ref, group_size=G)

    # Simulated Data: One prompt, G sampled responses
    # shape: (G, SEQ_LEN)
    sample_response_ids = torch.randint(0, VOCAB_SIZE, (G, SEQ_LEN))
    sample_mask = torch.ones((G, SEQ_LEN))
    # Simulated rewards for the group
    sample_rewards = torch.tensor([1.0, 2.5, 0.5, 4.0]) 

    print(f"Starting GRPO Training Step (G={G})...")
    
    # Training Step
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    optimizer.zero_grad()
    loss, metrics = trainer.compute_loss(
        prompt_ids=None, # Mock implementation doesn't use prompt_ids directly
        response_ids=sample_response_ids,
        attention_mask=sample_mask,
        rewards=sample_rewards
    )
    loss.backward()
    optimizer.step()
    
    trainer.log_step(metrics)

    print("Step Completed Successfully.")
    print(f"Metrics: {metrics}")
    print(f"Reward History: {trainer.history['reward']}")