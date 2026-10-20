"""CPU tests for the M4 screen harness (experiments/m4_routing_arms.py). No GPU, no Modal.

Needs torch + torchvision + zoology deps, and a zoology checkout (pin 1ad20d1) PATCHED with
experiments/patch_zoology_accum.py:
    ZOOLOGY_DIR=/path/to/zoology python experiments/test_m4_harness.py

zoology 1ad20d1 hard-codes device='cuda' for its embeddings; on CPU-only machines these tests
retarget that default to CPU (test-only; the harness itself is unchanged).

  H1 trainer equivalence -- the screen trainer's fit() produces bit-identical parameters and
     validation curves to zoology's own (patched) Trainer.fit, 3 epochs, grad-accum 2, two arms.
  H2 end-to-end run      -- every record file exists; train/validation/held-out share no sequence;
     result.json carries the primary metric.
  H3 time cap            -- stops early, flags time_capped, still reports the held-out result.
  H4 crash               -- a failure after setup still writes result.json (status=failed), exit != 0.
  H5 never overwrite     -- rerunning the same arm+seed creates a second folder.
  H6 unpatched zoology   -- refuses to start, creates no folder.
  H7 cache state         -- manifest records cold vs warm data caches correctly.
  H8 reproducible data   -- two cold generations (different arms) give identical train/val/held-out.
  H9 dense arm           -- zoology MHA (dropout 0.1, 4 heads) runs end-to-end; s1 recorded as None.
  H10 --s1 budget        -- default stays S1=2 (34 pairs, run id s2); --s1 12 reaches the mixer
     (config shows num_selected_superblocks=12, 144 pairs, run id s12); same data as S1=2.
"""
import glob, importlib.util, inspect, json, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "m4_routing_arms.py")
ZDIR = os.environ.get("ZOOLOGY_DIR") or os.path.join(REPO, "zoology")
os.environ.update(WANDB_MODE="disabled", GRAD_ACCUM="2", GRAD_CLIP="0", TQDM_DISABLE="1")
sys.path[:0] = [ZDIR, os.path.join(REPO, "src")]
import torch
FAILS = []
SHIM = ("import inspect,runpy,sys,zoology.model as zm\n"
        "for _,c in inspect.getmembers(zm, inspect.isclass):\n"
        "    d=getattr(c.__init__,'__defaults__',None)\n"
        "    if d and 'cuda' in d: c.__init__.__defaults__=tuple('cpu' if x=='cuda' else x for x in d)\n"
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')\n")

def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not cond: FAILS.append(name)

def cpu_retarget():
    if torch.cuda.is_available(): return
    import zoology.model as zm
    for _, c in inspect.getmembers(zm, inspect.isclass):
        d = getattr(c.__init__, "__defaults__", None)
        if d and "cuda" in d: c.__init__.__defaults__ = tuple("cpu" if x == "cuda" else x for x in d)

def run(out, cache, *args, zdir=ZDIR):
    env = {**os.environ, "ZOOLOGY_DIR": zdir, "PYTHONPATH": f"{zdir}{os.pathsep}{os.path.join(REPO, 'src')}"}
    base = [sys.executable] + (["-c", SHIM, SCRIPT] if not torch.cuda.is_available() else [SCRIPT])
    small = ["--n-train", "32", "--n-val", "16", "--n-test", "16", "--out-root", out, "--cache-dir", cache,
             "--device", "cuda" if torch.cuda.is_available() else "cpu"]
    return subprocess.run(base + small + list(args), env=env, capture_output=True, text=True)  # caller args win

def latest(out, arm):
    d = sorted(glob.glob(os.path.join(out, f"*-{arm}-seed*")))[-1]
    return d, json.load(open(os.path.join(d, "result.json"))), json.load(open(os.path.join(d, "manifest.json")))

def h1():
    print("H1 screen trainer == zoology Trainer (bit-identical)")
    cpu_retarget()
    import torch.nn as nn, torch.optim as optim
    from zoology.config import DataConfig, ModelConfig, ModuleConfig
    from zoology.data.multiquery_ar import MQARConfig
    from zoology.data.utils import prepare_data
    from zoology.model import LanguageModel
    from zoology.train import Trainer
    from zoology.utils import set_determinism
    spec = importlib.util.spec_from_file_location("m4", SCRIPT); m4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m4)
    class Log:
        def log(self, *a, **k): pass
    cache = tempfile.mkdtemp()
    mq = lambda n: MQARConfig(num_examples=n, vocab_size=8192, input_seq_len=256, num_kv_pairs=16)
    data = DataConfig(train_configs=[mq(64)], test_configs=[mq(16)], batch_size=8, seed=7, cache_dir=cache)
    prepare_data(data)   # warm the cache so both runs load identically (see H7/H8 for why)
    for arm in ("baseline", "router+maxpool"):
        mixer = ModuleConfig(name="hier_nsa.HierSparseAttention", kwargs=dict(num_heads=4, compress_block_size=16,
            selection_block_size=16, num_selected_blocks=4, superblock_group=4, num_selected_superblocks=2, **m4.ARMS[arm]))
        mcfg = ModelConfig(vocab_size=8192, max_position_embeddings=256, d_model=64, n_layers=2, sequence_mixer=mixer)
        states, curves = [], []
        for kind in ("stock", "screen"):
            set_determinism(0); model = LanguageModel(mcfg); tr, va = prepare_data(data); curve = []
            Cls = Trainer if kind == "stock" else m4.screen_trainer(Trainer, nn, optim, lambda t, e, m: None, lambda t, e: False)
            t = Cls(model=model, train_dataloader=tr, test_dataloader=va, max_epochs=3, learning_rate=3.2e-4,
                    weight_decay=0.1, early_stopping_metric="valid/accuracy", early_stopping_threshold=0.98,
                    device="cuda" if torch.cuda.is_available() else "cpu", logger=Log())
            orig = t.test
            t.test = lambda e, _o=orig: (lambda m: (curve.append(m["valid/accuracy"]), m)[1])(_o(e))
            t.fit(); states.append({k: v.clone() for k, v in model.state_dict().items()}); curves.append(curve)
        same = all(torch.equal(states[0][k], states[1][k]) for k in states[0])
        check(f"{arm}: {len(states[0])} parameter tensors identical, curves identical", same and curves[0] == curves[1])

def h2_to_h8():
    root = tempfile.mkdtemp(); out = os.path.join(root, "runs")
    print("H2 end-to-end run")
    p = run(out, os.path.join(root, "c1"), "--arm", "router", "--epochs", "1")
    d, r, m = latest(out, "router")
    files = {f for f in os.listdir(d)}
    check("exit 0 and all record files present", p.returncode == 0 and
          {"manifest.json", "metrics.jsonl", "result.json", "train.log", "best.pt", "final.pt"} <= files, str(sorted(files)))
    check("splits share no sequence", r.get("split_overlap") and sum(r["split_overlap"].values()) == 0, str(r.get("split_overlap")))
    check("primary metric reported", r.get("status") == "complete" and "PRIMARY_test_accuracy_at_best_validation" in r)
    print("H3 time cap")
    p = run(out, os.path.join(root, "c1"), "--arm", "baseline", "--epochs", "3", "--max-hours", "0.000001")
    _, r, _ = latest(out, "baseline")
    check("stopped after 1 of 3 epochs, flagged, still reported", p.returncode == 0 and r["epochs_run"] == 1 and
          r["time_capped"] and "PRIMARY_test_accuracy_at_best_validation" in r)
    print("H4 crash still writes result.json")
    p = run(out, os.path.join(root, "c1"), "--arm", "maxpool", "--epochs", "1", "--n-val", "0")
    _, r, _ = latest(out, "maxpool")
    check("exit != 0, status failed, error recorded", p.returncode != 0 and r["status"] == "failed" and r.get("error"), r.get("error", ""))
    print("H5 never overwrite")
    time.sleep(1.1); run(out, os.path.join(root, "c1"), "--arm", "baseline", "--epochs", "1")
    check("second folder for the same arm+seed", len(glob.glob(os.path.join(out, "*-baseline-seed*"))) == 2)
    print("H6 unpatched zoology refuses")
    unpatched = os.path.join(root, "zoology_unpatched"); shutil.copytree(ZDIR, unpatched)
    subprocess.run(["git", "checkout", "-q", "zoology/train.py"], cwd=unpatched)
    n0 = len(os.listdir(out)); p = run(out, os.path.join(root, "c1"), "--arm", "router", "--epochs", "1", zdir=unpatched)
    check("refused, no folder created", p.returncode != 0 and "not patched" in (p.stdout + p.stderr) and len(os.listdir(out)) == n0)
    print("H7 cache state recorded")
    run(out, os.path.join(root, "c2"), "--arm", "router+maxpool", "--epochs", "1"); time.sleep(1.1)
    run(out, os.path.join(root, "c2"), "--arm", "router+maxpool", "--epochs", "1")
    ms = [json.load(open(os.path.join(x, "manifest.json")))["data_cache_hit_at_start"]
          for x in sorted(glob.glob(os.path.join(out, "*-router+maxpool-seed*")))]
    check("cold run all False, warm run all True", not any(ms[0].values()) and all(ms[1].values()), str(ms))
    print("H8 cold generations reproducible across arms")
    a = {os.path.basename(x): torch.load(x) for x in glob.glob(os.path.join(root, "c1", "*.pt"))}
    b = {os.path.basename(x): torch.load(x) for x in glob.glob(os.path.join(root, "c2", "*.pt"))}
    same = [k for k in a if k in b and torch.equal(a[k]["inputs"], b[k]["inputs"]) and torch.equal(a[k]["labels"], b[k]["labels"])]
    check("train, validation and held-out identical (router vs router+maxpool cold caches)", len(same) >= 3, f"{len(same)} datasets matched")
    print("H9 dense arm")
    p = run(out, os.path.join(root, "c3"), "--arm", "dense", "--epochs", "1")
    d, r, m = latest(out, "dense")
    check("dense: complete, MHA mixer, s1 None, primary metric", p.returncode == 0 and r["status"] == "complete"
          and m["mixer"] == "zoology.mixers.attention.MHA" and m["arm_kwargs"] == {"dropout": 0.1, "num_heads": 4}
          and m["s1"] is None and m["protocol"]["coarse_pairs_per_query"] is None
          and "PRIMARY_test_accuracy_at_best_validation" in r and os.path.basename(d).startswith("m4-L2048-dense-seed123"),
          os.path.basename(d))
    print("H10 --s1 budget")
    _, _, m2 = latest(out, "s2-k4-baseline")
    check("default baseline is S1=2: 34 pairs, run id s2", m2["s1"] == 2 and m2["protocol"]["coarse_pairs_per_query"] == 34
          and m2["run_id"] == "m4-L2048-s2-k4-baseline-seed123" and m2["protocol_conformant"] is not None)
    p = run(out, os.path.join(root, "c3"), "--arm", "baseline", "--s1", "12", "--epochs", "1")
    d, r, m = latest(out, "s12-k4-baseline")   # by name: "s12" sorts before "s2"
    log = open(os.path.join(d, "train.log")).read()
    check("--s1 12 reaches the mixer: 144 pairs, run id s12, config num_selected_superblocks=12",
          p.returncode == 0 and r["status"] == "complete" and m["s1"] == 12
          and m["protocol"]["coarse_pairs_per_query"] == 144 and m["run_id"] == "m4-L2048-s12-k4-baseline-seed123"
          and "'num_selected_superblocks': 12" in log, m["run_id"])
    check("dense and S1=12 used the same (cached) data as the cold dense run",
          all(m["data_cache_hit_at_start"].values()), str(m["data_cache_hit_at_start"]))
    shutil.rmtree(root, ignore_errors=True)

if __name__ == "__main__":
    if "GRAD_CLIP" not in open(os.path.join(ZDIR, "zoology", "train.py")).read():
        sys.exit(f"ZOOLOGY_DIR={ZDIR} is missing or unpatched; run experiments/patch_zoology_accum.py on it first")
    h1(); h2_to_h8()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
