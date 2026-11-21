import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

class GRPOConfig:
    def __init__(
        self,
        num_train_epochs: int = 1,
        learning_rate: float = 5e-6,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 0.3,
        beta: float = 1.0,
        kl_coef: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam

class GRPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        reward_fn: callable,
        config: GRPOConfig,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.per_device_train_batch_size) if train_dataset else None
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=config.per_device_train_batch_size) if eval_dataset else None

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config.num_train_epochs
        )

        # Store old log probabilities for KL computation
        self.old_log_probs = None

    def compute_rewards(self, outputs: Dict[str, Any]) -> torch.Tensor:
        rewards = self.reward_fn(outputs)
        return rewards

    def compute_gae(self, rewards: torch.Tensor, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generalized Advantage Estimation"""
        if values is None:
            advantages = rewards - rewards.mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return advantages

        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def compute_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
        """Proper KL divergence computation between old and new policies"""
        kl = (old_log_probs - new_log_probs).exp() * (new_log_probs - old_log_probs)
        return kl.mean()

    def compute_loss(self, model_outputs: Dict[str, Any], rewards: torch.Tensor) -> torch.Tensor:
        log_probs = model_outputs["log_probs"]
        advantages = self.compute_gae(rewards)

        # GRPO loss with proper KL penalty
        policy_loss = -torch.mean(advantages * log_probs)
        kl_div = 0.0

        if self.old_log_probs is not None:
            kl_div = self.compute_kl_divergence(self.old_log_probs, log_probs)

        # Full GRPO loss: β*r + (1-β)*logπ - KL penalty
        loss = -torch.mean(self.config.beta * rewards + (1 - self.config.beta) * log_probs)
        loss += self.config.kl_coef * kl_div

        return loss

    def train(self):
        self.model.train()
        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                # Forward pass
                model_outputs = self.model(**batch)
                log_probs = model_outputs["log_probs"]

                # Store old log probabilities before computing loss
                if self.old_log_probs is None:
                    self.old_log_probs = torch.zeros_like(log_probs)

                # Compute rewards and loss
                rewards = self.compute_rewards(model_outputs)
                loss = self.compute_loss(model_outputs, rewards)

                # Backward pass
                loss.backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Update old log probabilities
                self.old_log_probs = log_probs.detach().clone()

            # Evaluation
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                print(f"Epoch {epoch}, Eval Metrics: {eval_metrics}")

        return self.model

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_rewards = 0
        count = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                model_outputs = self.model(**batch)
                rewards = self.compute_rewards(model_outputs)
                total_rewards += rewards.sum().item()
                count += rewards.numel()

        avg_reward = total_rewards / count if count > 0 else 0
        return {"avg_reward": avg_reward}

    def step(self, model_outputs: Dict[str, Any]):
        rewards = self.compute_rewards(model_outputs)
        loss = self.compute_loss(model_outputs, rewards)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return loss.item()

    def log(self, logs: Dict[str, Any]):
        print(logs)

# Example usage
if __name__ == "__main__":
    # Dummy model and dataset for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, input_ids):
            return {"log_probs": torch.randn(8, 5)}

    model = DummyModel()
    reward_fn = lambda x: torch.randn(8, 5)
    config = GRPOConfig()
    trainer = GRPOTrainer(model=model, reward_fn=reward_fn, config=config)
    trained_model = trainer.train()
