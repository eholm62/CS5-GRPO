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
        """A full training run on the custom trainer matches TRL's run exactly."""
        config_trl = self.make_config()
        config_custom = self.make_config()

        trl_kwargs = self.build_kwargs(TRL_GRPOTrainer, self.model_trl, config_trl)
        custom_kwargs = dict(trl_kwargs)
        custom_kwargs["model"] = self.model_custom
        custom_kwargs["args"] = config_custom

        trainer_trl = TRL_GRPOTrainer(**trl_kwargs)
        trainer_custom = Custom_GRPOTrainer(**custom_kwargs)

        reset_global_seeds()
        output_trl = trainer_trl.train()
        reset_global_seeds()
        output_custom = trainer_custom.train()

        # 1. The reported training loss must match.
        loss_trl = float(output_trl.train_loss)
        loss_custom = float(output_custom.train_loss)
        self.assertLessEqual(
            abs(loss_trl - loss_custom),
            1e-4,
            f"train_loss mismatch: TRL={loss_trl}, Custom={loss_custom}",
        )

        # 2. The logged per-step metrics must match.
        entries_trl = [entry for entry in trainer_trl.state.log_history if "loss" in entry]
        entries_custom = [entry for entry in trainer_custom.state.log_history if "loss" in entry]
        self.assertTrue(entries_trl, "TRL log_history contains no step entry with 'loss'")
        self.assertTrue(entries_custom, "Custom log_history contains no step entry with 'loss'")
        entry_trl, entry_custom = entries_trl[-1], entries_custom[-1]

        self.assertIn("loss", entry_trl, "TRL did not log 'loss'")
        self.assertIn("loss", entry_custom, "Custom did not log 'loss'")
        for key in ("reward", "kl"):
            if key in entry_trl:
                self.assertIn(key, entry_custom, f"Custom did not log '{key}' which TRL logged")

        # Compare the version-independent core metrics TRL logged.
        skip_keys = {"epoch", "step", "total_flos", "train_steps_per_second"}
        common_keys = (set(entry_trl) & set(entry_custom)) - skip_keys
        for key in sorted(common_keys & {"loss", "reward"}):
            value_trl, value_custom = entry_trl[key], entry_custom[key]
            if not isinstance(value_trl, (int, float)) or isinstance(value_trl, bool):
                continue
            self.assertLessEqual(
                abs(value_trl - value_custom),
                1e-4 * max(1.0, abs(value_trl)),
                f"metric '{key}' mismatch: TRL={value_trl}, Custom={value_custom}",
            )

        # 3. The final weights must match.
        trl_params = {name: param.detach() for name, param in trainer_trl.model.named_parameters()}
        for name, param_custom in trainer_custom.model.named_parameters():
            self.assertIn(name, trl_params, f"custom parameter '{name}' missing from TRL model")
            self.assertTrue(
                torch.allclose(trl_params[name], param_custom.detach(), atol=1e-5),
                f"weight '{name}' diverged between TRL and custom runs",
            )

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