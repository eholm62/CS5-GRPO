import unittest
import torch
import torch.nn as nn
from implementation6 import GRPOSTrainer

class MockModel(nn.Module):
    """A simple mock model to satisfy the GRPOSTrainer requirements."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, input_ids, attention_mask=None):
        # Returns a mock output object with a 'logits' attribute
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        
        # Create dummy logits matching the shape of input_ids
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 10)
        return MockOutput(logits)

class TestGRPOSTrainer(unittest.TestCase):
    def setUp(self):
        self.policy = MockModel()
        self.ref_model = MockModel()
        self.trainer = GRPOSTrainer(
            policy_model=self.policy,
            ref_model=self.ref_model,
            group_size=8
        )

    def test_compute_group_advantages_normalization(self):
        """
        Tests that advantage normalization results in a mean of approximately 0 
        and a standard deviation of approximately 1.
        """
        # Create a dummy reward tensor for a group (e.g., size 8)
        # We use a non-uniform distribution to ensure normalization is actually working
        rewards = torch.tensor([1.0, 2.0, 5.0, 10.0, 0.5, 3.0, 7.0, 4.0], dtype=torch.float32)
        
        # Compute advantages
        advantages = self.trainer.compute_group_advantages(rewards)
        
        # Check mean is approximately 0
        mean_val = advantages.mean().item()
        self.assertAlmostEqual(mean_val, 0.0, places=5, 
                               msg=f"Mean of advantages should be ~0, got {mean_val}")
        
        # Check standard deviation is approximately 1
        std_val = advantages.std().item()
        self.assertAlmostEqual(std_val, 1.0, places=5, 
                               msg=f"Std of advantages should be ~1, got {std_val}")

    def test_compute_group_advantages_with_eps_stable(self):
        """
        Tests that the epsilon stability prevents division by zero 
        when all rewards are identical.
        """
        # All rewards are the same, so std would be 0
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float32)
        
        # This should not crash due to self.eps_stable in compute_group_advantages
        try:
            advantages = self.trainer.compute_group_advantages(rewards)
        except ZeroDivisionError:
            self.fail("compute_group_advantages raised ZeroDivisionError!")
            
        # With identical rewards, mean is 5, std is 0. 
        # (rewards - mean) is [0, 0, 0, 0]. 
        # Result should be 0/eps_stable = 0.
        self.assertTrue(torch.all(advantages == 0))

    def test_advantage_shape(self):
        """Ensures the output shape of advantages matches input shape."""
        rewards = torch.randn(10)
        advantages = self.trainer.compute_group_advantages(rewards)
        self.assertEqual(advantages.shape, rewards.shape)

if __name__ == "__main__":
    unittest.main()
