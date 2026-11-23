# debug_compare.py
import inspect, random, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer as TRLTrainer
from grpo import GRPOTrainer as MyTrainer

SEED, MODEL = 123, "Qwen/Qwen2.5-0.5B"
SEED, MODEL = 123, "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
data = Dataset.from_list([{"prompt": p} for p in ["Hi", "Say hello to me please", "Write a poem about the ocean and the stars", "x"]])

REWARDS = []
def reward(prompts, completions, **kw):
    s = [float(len(c)) for c in completions]
    REWARDS.extend(s); return s

def mkcfg():
    cands = dict(output_dir="/tmp/x", num_generations=2, per_device_train_batch_size=4,
                 learning_rate=1e-3, beta=0.0, report_to=[], max_steps=1,
                 seed=SEED, data_seed=SEED,
                 max_length=48, max_prompt_length=32, max_completion_length=16)
    ok = set(inspect.signature(GRPOConfig).parameters)
    kwargs = {k: v for k, v in cands.items() if k in ok}
    return GRPOConfig(**kwargs)

def build(m, cls):
    return cls(model=m, reward_funcs=[reward], args=mkcfg(),
               train_dataset=data, processing_class=tok)

def run(cls):
    REWARDS.clear(); GEN = {}
    m = AutoModelForCausalLM.from_pretrained(MODEL)
    torch.manual_seed(SEED); random.seed(SEED)
    t = build(m, cls)
    orig = m.generate
    def hook(*a, **k):
        inp = a[0] if a else k.get("input_ids"); outp = orig(*a, **k)
        GEN["plen"] = inp.shape[1]; GEN["out"] = outp; return outp
    m.generate = hook
    init = [p.detach().cpu().clone() for p in m.parameters()]
    res = t.train()
    max_diff = torch.stack([(p.detach().cpu() - q).abs().max() for p, q in zip(m.parameters(), init)]).max().item()
    changed = bool(max_diff > 0 or max_diff != max_diff)  # NaN counts as changed
    print(f"{cls.__module__}.{cls.__name__}: max_diff={max_diff:.6e} changed={changed}")
    comps = GEN["out"][:, GEN["plen"]:].cpu() if "out" in GEN else None
    return dict(comps=comps, rewards=list(REWARDS), loss=float(res.training_loss), changed=changed)

A = run(TRLTrainer); B = run(MyTrainer)
def show(n, x, y): print(f"{n:14} | TRL : {x}\n{'':14} | MINE: {y}\n")
show("completions", A["comps"], B["comps"])
show("rewards", A["rewards"], B["rewards"])
show("loss", A["loss"], B["loss"])
show("weights", A["changed"], B["changed"])
