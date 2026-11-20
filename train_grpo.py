import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import logging
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

# Importing the provided core logic
from implementation6 import GRPOSTrainer

# Setup logging for convergence monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CharacterCountingDataset(Dataset):
    """
    Parses CSV data and formats it into natural language prompts.
    Expected CSV columns: text_used, character, target_count, difficulty_level
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.prompts = []
        self.targets = []
        
        for _, row in self.df.iterrows():
            prompt = f"How many times does the character '{row['character']}' appear in '{row['text_used']}'?"
            self.prompts.append(prompt)
            self.targets.append(int(row['target_count']))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "target": self.targets[idx]
        }

class ToyLanguageModel(nn.Module):
    """
    A tiny GPT-style mock model for demonstration purposes.
    In a real scenario, this would be a loaded Transformer model.
    """
    def __init__(self, vocab_size=100, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return type('Output', (object,), {'logits': logits})

class TrainingEngine:
    def __init__(
        self, 
        policy: nn.Module, 
        ref_model: nn.Module, 
        trainer: GRPOSTrainer, 
        device: str
    ):
        self.policy = policy.to(device)
        self.ref_model = ref_model.to(device)
        self.trainer = trainer
        self.device = device
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-4)

    def rule_based_reward(self, completions: List[str], targets: List[int]) -> torch.Tensor:
        """
        Implements the reward logic:
        Reward = 1 if output matches target_count exactly.
        Reward = 0 otherwise.
        """
        rewards = []
        for text, target in zip(completions, targets):
            # Extract digit using regex in case model outputs "The answer is 3"
            match = re.search(r'\d+', text)
            if match and int(match.group()) == target:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards, dtype=torch.float32).to(self.device)

    def generate_group_completions(self, prompts: List[str], group_size: int) -> Dict[str, torch.Tensor]:
        """
        Simulates generating multiple completions per prompt for GRPO.
        In production, this would use model.generate() with temperature/top-p.
        """
        batch_size = len(prompts)
        # Placeholder: simulating tokenized input and output shapes
        # (batch_size * group_size, seq_len)
        total_samples = batch_size * group_size
        
        # For this toy example, we create dummy tensors representing sampled IDs
        # In a real setup, this uses the policy model to sample tokens
        input_ids = torch.randint(0, 100, (total_samples, 16)).to(self.device)
        attention_mask = torch.ones((total_samples, 16)).to(self.device)
        
        # Mocking 'old_log_probs' which are required by the trainer for importance sampling [1]
        old_log_probs = torch.randn(total_samples, 16).to(self.device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "old_log_probs": old_log_probs,
            "text_completions": ["3"] * total_samples # Mocking strings for reward function
        }

    def train_step(self, batch: Dict[str, Any], target_counts: List[int]):
        self.optimizer.zero_grad()
        
        # 1. Generate Grouped Samples
        # GRPO requires G samples per prompt to compute relative advantages [1]
        samples = self.generate_group_completions(batch['prompt'], self.trainer.group_size)
        
        # 2. Compute Rewards
        # Flatten targets to match the (Batch * Group) shape
        expanded_targets = []
        for t in target_counts:
            expanded_targets.extend([t] * self.trainer.group_size)
            
        rewards = self.rule_based_reward(samples['text_completions'], expanded_targets)
        
        # 3. Compute Loss via Trainer
        # The trainer uses importance sampling: ratio = exp(current_log_probs - old_log_probs) [1]
        loss, metrics = self.trainer.compute_loss(
            response_ids=samples['input_ids'],
            attention_mask=samples['attention_mask'],
            rewards=rewards,
            old_log_probs=samples['old_log_probs']
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), metrics, rewards.mean().item()

def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CSV_PATH = "training_data.csv"
    EPOCHS = 5
    BATCH_SIZE = 4 # Number of unique prompts per batch
    
    # 0. Create dummy data if not exists
    import os
    if not os.path.exists(CSV_PATH):
        pd.DataFrame({
            'text_used': ['banana intelligence hippopotamus', 'apple pie', 'test'],
            'character': ['i', 'p', 't'],
            'target_count': [3, 2, 1],
            'difficulty_level': [1, 1, 1]
        }).to_csv(CSV_PATH, index=False)

    # 1. Initialize Models and Trainer
    policy = ToyLanguageModel()
    ref_model = ToyLanguageModel()
    trainer = GRPOSTrainer(
        policy_model=policy,
        ref_model=ref_model,
        group_size=4 # G=4
    )
    
    engine = TrainingEngine(policy, ref_model, trainer, DEVICE)
    dataset = CharacterCountingDataset(CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Training Loop
    logger.info("Starting GRPO Training Loop...")
    history = {"loss": [], "reward": [], "kl": []}

    for epoch in range(EPOCHS):
        epoch_loss, epoch_reward, epoch_kl = 0, 0, 0
        
        for i, batch in enumerate(dataloader):
            # Convert batch to device
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # For text prompts, we'd normally tokenize them here. 
            # For this toy script, we simulate the list of strings.
            
            targets = batch['target'].tolist()
            
            loss, metrics, avg_reward = engine.train_step(batch, targets)
            
            # Logging
            epoch_loss += loss
            epoch_reward += avg_reward
            epoch_kl += metrics.get('kl', 0.0)
            
            logger.info(f"Epoch {epoch+1} | Batch {i+1} | Loss: {loss:.4f} | Avg Reward: {avg_reward:.4f}")

        # Summary for Epoch
        avg_loss = epoch_loss / len(dataloader)
        avg_rew = epoch_reward / len(dataloader)
        avg_kl = epoch_kl / len(dataloader)
        
        history["loss"].append(avg_loss)
        history["reward"].append(avg_rew)
        history["kl"].append(avg_kl)
        
        logger.info(f"--- Epoch {epoch+1} Summary ---")
        logger.info(f"Avg Loss: {avg_loss:.4f} | Avg Reward: {avg_rew:.4f} | Avg KL: {avg_kl:.4f}")

    # 3. Final Convergence Monitoring
    logger.info("========================================")
    logger.info("TRAINING COMPLETE")
    final_reward = history["reward"][-1]
    logger.info(f"Final Average Reward: {final_reward:.4f}")
    
    if final_reward > 0.7:
        logger.info("Status: SUCCESS - Model has converged on the task.")
    else:
        logger.info("Status: FAILURE - Reward too low. Consider increasing training steps or group size.")
    logger.info("========================================")

if __name__ == "__main__":
    main()