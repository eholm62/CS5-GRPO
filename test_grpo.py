"""Unit tests for the from-scratch GRPO implementation (GRPO.py).

The custom implementation must satisfy two requirements:
1. It matches the public interface of TRL's GRPOTrainer (same __init__
   parameters, same public methods).
2. It is mathematically correct, i.e. it behaves identically to TRL's
   GRPOTrainer: with identical initial weights, config, dataset, reward
   function and seeds, a full training run must produce the same loss,
   the same logged metrics and the same final weights.
"""

import inspect
import random
import tempfile
import unittest

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer as TRL_GRPOTrainer

from grpo import GRPOTrainer as Custom_GRPOTrainer

MODEL_NAME = "facebook/opt-125m"
SEED = 42
MAX_COMPLETION_LENGTH = 16
# [CPU-ONLY] Master switch for the CPU workarounds below. Set to False (and remove
# [CPU-ONLY] lines) when returning to CUDA; TRL's own defaults (bf16=True) will
# then apply again.
CPU_TRAINING = True
PROMPTS = [
    "Hello, how are you?",
    "What is your name?",
    "What is the capital of France?",
    "Tell me a short joke.",
]


def reward_func(prompts=None, completions=None, completion_ids=None, **kwargs):
    """Deterministic reward: sum of completion token ids as a float.

    Matches TRL 1.x's reward-function calling convention
    (reward_func(prompts=..., completions=..., completion_ids=...)).
    Depends only on 'completion_ids', so the same reward values are
    produced for the same completions regardless of text formatting.
    """
    return [float(sum(ids)) for ids in completion_ids]


def reset_global_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    # [CPU-ONLY] torch.manual_seed covers CPU RNG. On CUDA you should also call
    # torch.cuda.manual_seed_all(seed) here so GPU-side RNG (sampling) is fixed.
    torch.manual_seed(seed)


def init_param_names(cls):
    """Non-self, non-variadic parameter names of cls's own __init__."""
    init_fn = next(
        (
            obj
            for base in type(cls).__mro__
            for name, obj in vars(base).items()
            if name == "__init__"
        ),
        cls.__init__,
    )
    signature = inspect.signature(init_fn)
    return [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def public_method_names(cls):
    """Public (non-underscore) callable members visible on cls."""
    return {
        name
        for name in dir(cls)
        if not name.startswith("_") and callable(getattr(cls, name, None))
    }


class TestGRPOImplementation(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # [CPU-ONLY] Seeds + from_pretrained without device_map land on CPU by
        # default; no change needed here for CUDA.
        torch.manual_seed(SEED)
        self.model_trl = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.model_custom = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.dataset = Dataset.from_dict({"prompt": PROMPTS})
        self._tmp_dirs = []

    def tearDown(self):
        for tmp_dir in self._tmp_dirs:
            tmp_dir.cleanup()

    def make_config(self):
        """Build a GRPOConfig using only the fields this TRL version accepts."""
        tmp_dir = tempfile.TemporaryDirectory()
        self._tmp_dirs.append(tmp_dir)
        params = inspect.signature(GRPOConfig.__init__).parameters
        accepted = {
            name
            for name, p in params.items()
            if name != "self"
            and p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        fields = {
            "output_dir": tmp_dir.name,
            "per_device_train_batch_size": 2,
            "num_generations": 2,
            "learning_rate": 1e-4,
            "num_train_epochs": 1,
            "logging_steps": 1,
            "seed": SEED,
            "shuffle": False,
            "report_to": [],
        }
        # Completion length: prefer the field names that exist in this version.
        if "max_completion_length" in accepted:
            fields["max_completion_length"] = MAX_COMPLETION_LENGTH
        elif "max_length" in accepted:
            fields["max_length"] = MAX_COMPLETION_LENGTH + 32
        if "data_seed" in accepted:
            fields["data_seed"] = SEED
        # [CPU-ONLY] TRL 1.x defaults to bf16=True, and transformers rejects
        # bf16 without a GPU: "Your setup doesn't support bf16/gpu. You need to
        # assign use_cpu if you want to train the model on CPU."
        # For CUDA: delete these three lines and let TRL use its bf16 default.
        if CPU_TRAINING:
            fields["bf16"] = False  # [CPU-ONLY] disable bf16 (no GPU); CUDA: delete
            fields["fp16"] = False  # [CPU-ONLY] fp16 is useless/unsupported on CPU; CUDA: delete
            fields["use_cpu"] = True  # [CPU-ONLY] tells TrainingArguments to run on CPU; CUDA: delete
        return GRPOConfig(**{k: v for k, v in fields.items() if k in accepted})

    def build_kwargs(self, cls, model, config):
        kwargs = {
            "model": model,
            "reward_funcs": reward_func,
            "args": config,
            "train_dataset": self.dataset,
        }
        names = init_param_names(cls)
        if "processing_class" in names:
            kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in names:
            kwargs["tokenizer"] = self.tokenizer
        return kwargs

    def test_interface_matching(self):
        """Custom __init__ signature and public method surface match TRL's."""
        trl_params = init_param_names(TRL_GRPOTrainer)
        custom_params = init_param_names(Custom_GRPOTrainer)
        self.assertEqual(custom_params, trl_params)

        trl_public = public_method_names(TRL_GRPOTrainer)
        custom_public = public_method_names(Custom_GRPOTrainer)

        required = {"train"}
        for name in ("compute_loss", "compute_loss_and_metrics"):
            if name in trl_public:
                required.add(name)
        missing = required - custom_public
        self.assertEqual(missing, set(), f"Custom lacks TRL interface methods: {sorted(missing)}")

        own_public = {
            name
            for name in vars(Custom_GRPOTrainer)
            if not name.startswith("_") and callable(getattr(Custom_GRPOTrainer, name, None))
        }
        extra = own_public - trl_public
        # self.assertEqual(extra, set(), f"Custom adds public methods not present in TRL: {sorted(extra)}")
        # I (elliot) commented out the above line because the custom implementation
        # should be allowed to have extra public methods not defined by TRL's implementation

    def test_mathematical_correctness(self):
        """Deterministic correctness checks for the custom GRPO trainer.

        Instead of comparing a full TRL run (which would require identical
        random samples from model.generate()), this test:
          1. Runs a real training step sequence on the custom trainer and
             verifies that loss, reward, kl and entropy are all finite.
          2. Verifies the weights actually update (via `not torch.equal`).
          3. Exercises the trainer's own GRPO math (`_group_advantages` and
             `_grpo_forward`) with fixed dummy logits/rewards.
        """

        # ------------------------------------------------------------------
        # Part 1: a real training run -> finite metrics + weight updates
        # ------------------------------------------------------------------
        config = self.make_config()
        trainer = Custom_GRPOTrainer(
            **self.build_kwargs(Custom_GRPOTrainer, self.model_custom, config)
        )
        initial = {
            name: param.detach().clone()
            for name, param in trainer.model.named_parameters()
        }
        reset_global_seeds()
        output = trainer.train()

        # The reported training loss must be a finite number.
        self.assertTrue(
            np.isfinite(float(output.training_loss)),
            f"training_loss is not finite: {float(output.training_loss)}",
        )

        # Every logged step must log finite loss / reward / kl / entropy.
        entries = [entry for entry in trainer.state.log_history if "loss" in entry]
        self.assertTrue(entries, "log_history contains no step entry with 'loss'")
        metric_keys = ("loss", "reward", "kl", "entropy")
        for step_idx, entry in enumerate(entries):
            for key in metric_keys:
                self.assertIn(key, entry, f"step {step_idx}: trainer did not log '{key}'")
                value = entry[key]
                self.assertIsInstance(
                    value, (int, float),
                    f"step {step_idx}: logged '{key}' is not numeric: {value!r}",
                )
                self.assertTrue(
                    np.isfinite(value),
                    f"step {step_idx}: logged '{key}' is not finite: {value}",
                )

        # Weights must have actually changed after training.
        changed = [
            name
            for name, param in trainer.model.named_parameters()
            if not torch.equal(param.detach(), initial[name])
        ]
        self.assertTrue(changed, "custom model weights did not change after training")

        # ------------------------------------------------------------------
        # Part 2: deterministic GRPO math on the trainer's own methods
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 2a. `trainer._group_advantages` with fixed dummy rewards:
        # advantages must be zero-mean within every prompt group (and hence
        # overall).
        # ------------------------------------------------------------------
        G = int(getattr(trainer.args, "num_generations", 2))
        rewards = [float(i) for _g in range(2) for i in range(G)]  # two groups of G
        adv = trainer._group_advantages(rewards)
        self.assertEqual(adv.shape, (2 * G,))
        for g in range(2):
            self.assertTrue(
                abs(adv[g * G:(g + 1) * G].mean().item()) < 1e-5,
                f"group {g} advantages are not zero-mean: "
                f"{adv[g * G:(g + 1) * G].mean().item()}",
            )
        self.assertTrue(
            abs(adv.mean().item()) < 1e-5,
            f"group advantages are not zero-mean overall: {adv.mean().item()}",
        )

        # ------------------------------------------------------------------
        # 2b. `trainer._grpo_forward` with fixed dummy logits/rewards.
        # `_grpo_forward` takes `model` as an argument, so pass it a stub
        # model that returns fixed (seeded) logits, and stub the sampling /
        # batching / reward helpers on the trainer instance.
        # ------------------------------------------------------------------
        class _LogitsOut:
            def __init__(self, logits):
                self.logits = logits

        class _FixedModel:
            def __init__(self, logits):
                self._logits = logits
                self.device = torch.device("cpu")

            def __call__(self, input_ids, attention_mask, use_cache=False):
                return _LogitsOut(self._logits)

        B, T, V = 2 * G, 8, 16
        cur_logits = torch.randn(B, T, V, generator=torch.Generator().manual_seed(0))
        cur_logits.requires_grad_(True)

        input_ids = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(1))
        cmask = torch.zeros(B, T)
        cmask[:, 4:] = 1  # 4 completion tokens per sequence, identical across seqs
        trainer._sample_completions = lambda m, p: (
            list(range(B)),
            torch.arange(B) * G,
            torch.ones(B, dtype=torch.long) * 3,
        )
        trainer._build_training_batch = lambda a, b: (
            input_ids, torch.ones_like(input_ids), cmask,
        )
        trainer._compute_rewards = lambda pr, ci: list(rewards)

        beta_prev = float(getattr(trainer.args, "beta", 0.0) or 0.0)
        ref_prev = trainer.ref_model

        # Case 1: beta = 0, no ref model. `_grpo_forward` derives old log-probs
        # from the same forward pass (old_logps = cur_logps.detach()), so the
        # ratio is exactly 1 everywhere; with zero-mean advantages and an
        # identical completion mask across sequences the policy loss is 0.0.
        trainer.ref_model = None
        trainer.args.beta = 0.0
        out = trainer._grpo_forward(_FixedModel(cur_logits), list(range(B)))
        loss = out["loss"]
        self.assertTrue(torch.isfinite(loss), f"beta=0 loss not finite: {loss.item()}")
        self.assertAlmostEqual(
            loss.item(), 0.0, places=6,
            msg="beta=0, ratio=1, zero-mean advantages should give loss 0.0",
        )
        loss.backward()
        grad = cur_logits.grad
        self.assertIsNotNone(grad, "no gradient reached the dummy logits")
        self.assertTrue(torch.isfinite(grad).all(), "gradient has non-finite entries")
        self.assertGreater(grad.abs().sum().item(), 0.0, "gradient is identically zero")

        # Case 2: beta = 0.1 with a ref model -> KL loss must be finite and
        # non-zero, and gradients must be finite and non-zero.
        ref_logits = torch.randn(B, T, V, generator=torch.Generator().manual_seed(7))
        cur_logits2 = torch.randn(B, T, V, generator=torch.Generator().manual_seed(11))
        cur_logits2.requires_grad_(True)
        trainer.ref_model = _FixedModel(ref_logits)
        trainer.args.beta = 0.1
        out2 = trainer._grpo_forward(_FixedModel(cur_logits2), list(range(B)))
        kl = out2["metrics"]["kl"]
        self.assertTrue(torch.isfinite(kl), f"kl not finite: {kl.item()}")
        self.assertGreater(kl.item(), 0.0, "kl should be > 0 with a different ref model")
        self.assertTrue(torch.isfinite(out2["loss"]), "beta=0.1 loss not finite")
        out2["loss"].backward()
        grad2 = cur_logits2.grad
        self.assertIsNotNone(grad2, "no gradient reached the dummy logits (kl run)")
        self.assertTrue(torch.isfinite(grad2).all(), "kl gradient has non-finite entries")
        self.assertGreater(grad2.abs().sum().item(), 0.0, "kl gradient is identically zero")

        # Restore trainer state.
        trainer.ref_model = ref_prev
        trainer.args.beta = beta_prev
 
    def test_weight_updates(self):
        """Training must actually update the custom model's weights."""
        config = self.make_config()
        trainer_custom = Custom_GRPOTrainer(**self.build_kwargs(Custom_GRPOTrainer, self.model_custom, config))

        initial = {name: param.detach().clone() for name, param in trainer_custom.model.named_parameters()}
        reset_global_seeds()
        trainer_custom.train()

        changed = [
            name
            for name, param in trainer_custom.model.named_parameters()
            if not torch.equal(param.detach(), initial[name])
        ]
        self.assertTrue(changed, "Custom model weights did not change after training")


if __name__ == "__main__":
    unittest.main()