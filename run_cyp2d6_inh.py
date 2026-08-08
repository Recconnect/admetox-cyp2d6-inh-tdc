"""End-to-end TDC CYP2D6_Veith (CYP2D6 inhibition, AUPRC) submission.

SOTA = 0.790 (CatBoost + MapLight 2580d + GIN embeddings, "NovoExpert-2").
This submission beats it: frozen per-seed leg evaluate_many = 0.821 +/- 0.001.

The reported protocol trains five independent ensemble runs. Each run averages
a disjoint group of three CatBoost seeds trained on MapLight 2580d features +
1200-d frozen Hu2020 GIN embeddings (4 x 300-d). TDC ``evaluate_many`` receives
the five distinct prediction vectors.

Modes:
  reproduce (default): deterministic. Rebuilds the five runs from the committed
      per-seed predictions in ``assets/cyp2d6_inh_legs.npz``, calls TDC
      ``group.evaluate_many`` and writes ``output/cyp2d6_inh_results.json``.
      No models are trained, no randomness is used.
  train: fresh end-to-end retraining. Recomputes MapLight features, extracts
      frozen GIN embeddings from the committed pretrained weights, trains the
      15 CatBoost seeds and rebuilds the five runs.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import average_precision_score
from tdc.benchmark_group import admet_group

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
ASSETS_DIR = ROOT / "assets"
GIN_WEIGHTS_DIR = ASSETS_DIR / "gin_weights"
ENDPOINT = "CYP2D6_Veith"
SOTA = 0.790
SEEDS_PER_RUN = 3
SEEDS = list(range(1, 16))


def log(message):
    print(message, flush=True)


def duration(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def seed_groups(seeds=None, per_run=None):
    seeds = seeds or SEEDS
    per_run = per_run or SEEDS_PER_RUN
    return [seeds[i:i + per_run] for i in range(0, len(seeds), per_run)]


def dataset_digest(train_smiles, train_y, test_smiles):
    payload = json.dumps(
        {"train_smiles": train_smiles, "train_y": train_y.tolist(), "test_smiles": test_smiles},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def environment_manifest():
    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": __import__("sklearn").__version__,
    }
    try:
        from rdkit import rdBase
        manifest["rdkit"] = rdBase.rdkitVersion
    except Exception:
        pass
    try:
        import catboost as cb
        manifest["catboost"] = cb.__version__
    except Exception:
        pass
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="TDC CYP2D6_Veith (AUPRC) submission")
    parser.add_argument("--mode", choices=["reproduce", "train"], default="reproduce",
                        help="reproduce: deterministic frozen leg (default); train: fresh end-to-end training")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--ss", type=float, default=0.6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated CatBoost seeds (default: 1-15)")
    parser.add_argument("--bs", type=int, default=1024, help="embedding batch size")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--quick", action="store_true",
                        help="installation smoke test; not the reported protocol")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.quick:
        args.iterations = min(args.iterations, 5)
        args.seeds = list(range(1, 6))
        args.seeds_per_run = 1
    else:
        args.seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(SEEDS)
        args.seeds_per_run = SEEDS_PER_RUN
    if len(args.seeds) < 5:
        raise ValueError("TDC evaluate_many requires at least five independent runs")
    if args.runs < 5:
        raise ValueError("TDC requires at least five independent runs")
    return args


def load_benchmark():
    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get(ENDPOINT)
    return group, benchmark, benchmark["name"]


def run_reproduce(args):
    """Deterministic reproduction of the frozen leg.

    Five independent runs are built from the committed per-seed predictions
    (``assets/cyp2d6_inh_legs.npz``): each run is the mean of a disjoint
    3-seed group. Expected result: evaluate_many = 0.821 +/- 0.001 (unrounded
    mean 0.8208, std 0.0013), all five vectors distinct.
    """
    started = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    legs_path = ASSETS_DIR / "cyp2d6_inh_legs.npz"
    if not legs_path.exists():
        raise FileNotFoundError(
            f"Missing {legs_path}; reproduce mode requires committed per-seed predictions")

    legs = np.load(legs_path)
    y_test = legs["y_test"].astype(np.int64)
    preds = legs["preds"].astype(np.float64)
    seeds = legs["seeds"].astype(np.int64)

    group, benchmark, name = load_benchmark()
    bench_y = benchmark["test"]["Y"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_test, bench_y):
        raise RuntimeError("assets/cyp2d6_inh_legs.npz labels do not match the TDC benchmark test set")
    train_val = benchmark["train_val"]
    digest = dataset_digest(
        train_val["Drug"].tolist(),
        train_val["Y"].to_numpy(dtype=int),
        benchmark["test"]["Drug"].tolist(),
    )

    log("=" * 72)
    log("ADMETox.AI: TDC CYP2D6_Veith (reproduce mode)")
    log(f"Frozen leg: {preds.shape[0]} CatBoost seeds x {preds.shape[1]} test molecules")
    log(f"Independent runs: {args.runs}; each = mean of {SEEDS_PER_RUN} disjoint seeds")
    log("=" * 72)

    groups = seed_groups()
    predictions_list, run_records, run_vectors = [], [], []
    for run_index, group_seeds in enumerate(groups, 1):
        mask = np.isin(seeds, group_seeds)
        vector = preds[mask].mean(axis=0)
        auprc = float(average_precision_score(y_test, vector))
        predictions_list.append({name: vector})
        run_vectors.append(vector)
        run_records.append({"run": run_index, "seeds": group_seeds, "auprc": auprc})
        log(f"Run {run_index}: AUPRC={auprc:.4f}; seeds={group_seeds} (frozen legs)")

    if len({vector.tobytes() for vector in run_vectors}) != len(run_vectors):
        raise RuntimeError("Independent runs produced duplicate prediction vectors")

    tdc_raw = group.evaluate_many(predictions_list)
    tdc_mean, tdc_std = [float(value) for value in tdc_raw[name]]
    run_scores = [record["auprc"] for record in run_records]
    run_mean = float(np.mean(run_scores))
    run_std = float(np.std(run_scores))
    result = {
        "endpoint": name,
        "metric": "AUPRC",
        "split": "official TDC scaffold split",
        "sota": SOTA,
        "protocol": ("frozen leg: 5 independent runs, each = mean of 3 disjoint CatBoost seeds "
                     "(MapLight 2580d + 4x300d frozen Hu2020 GIN embeddings, iter=800, lr=0.1, ss=0.6, cw=sqrt)"),
        "mode": "reproduce",
        "runs": run_records,
        "individual_auprcs": run_scores,
        "run_mean_auprc": run_mean,
        "run_std_auprc": run_std,
        "tdc_evaluate_many": {"mean": tdc_mean, "std": tdc_std, "note": "TDC rounds mean/std to 3 decimals"},
        "gap_to_sota": tdc_mean - SOTA,
        "beat_sota": tdc_mean > SOTA,
        "dataset_sha256": digest,
        "environment": environment_manifest(),
        "runtime_seconds": time.time() - started,
        "recorded_at": datetime.now().isoformat(),
    }
    result_name = "cyp2d6_inh_smoke_results.json" if args.quick else "cyp2d6_inh_results.json"
    prediction_name = "cyp2d6_inh_smoke_predictions.npz" if args.quick else "cyp2d6_inh_predictions.npz"
    with open(OUTPUT_DIR / result_name, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        OUTPUT_DIR / prediction_name,
        y_test=y_test,
        predictions=np.asarray(run_vectors),
    )

    log("=" * 72)
    log(f"TDC evaluate_many: {tdc_mean:.4f} +/- {tdc_std:.4f}")
    log(f"Gap to SOTA {SOTA:.3f}: {tdc_mean - SOTA:+.4f}")
    log(f"Distinct prediction vectors: {len(run_vectors)}")
    log(f"Runtime: {duration(time.time() - started)}")
    log(f"Saved: {OUTPUT_DIR / result_name}")
    log("=" * 72)


def train_mode(args):
    import catboost as cb
    import torch
    from rdkit import Chem, RDLogger

    from gin_pure import GraphBank, mean_pool, state_to_pure

    for channel in ["rdApp.info", "rdApp.warning", "rdApp.error", "rdApp.debug"]:
        RDLogger.DisableLog(channel)

    started = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    log("=" * 72)
    log("ADMETox.AI: TDC CYP2D6_Veith (train mode)")
    log(f"Independent runs: {args.runs}; seeds/run: {SEEDS_PER_RUN}; CatBoost iter={args.iterations}")
    if args.quick:
        log("QUICK SMOKE MODE: result is not the reported leaderboard protocol")
    log("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        with torch.no_grad():
            a = torch.randn(256, 256, device=device)
            for _ in range(3):
                (a @ a).sum().item()

    group, benchmark, name = load_benchmark()
    train_val, test = benchmark["train_val"], benchmark["test"]
    train_smiles = train_val["Drug"].tolist()
    test_smiles = test["Drug"].tolist()
    train_y = train_val["Y"].to_numpy(dtype=int)
    test_y = test["Y"].to_numpy(dtype=int)
    digest = dataset_digest(train_smiles, train_y, test_smiles)
    log(f"Dataset: train_val={len(train_y)}, test={len(test_y)}, hash={digest[:16]}")

    X_tv, X_te = build_features(train_smiles, test_smiles)
    log(f"MapLight features: {X_tv.shape[1]}d (2580 + 1200 frozen GIN embeddings)")

    preds = {}
    for index, seed in enumerate(args.seeds, 1):
        log(f"  CatBoost {index}/{len(args.seeds)}: seed {seed}")
        preds[seed] = train_catboost_seed(X_tv, train_y, X_te, seed, args, digest)

    groups = seed_groups(args.seeds, args.seeds_per_run)
    run_vectors = []
    for group_seeds in groups:
        run_vectors.append(np.mean([preds[s] for s in group_seeds], axis=0))

    predictions_list = [{name: v} for v in run_vectors]
    tdc_raw = group.evaluate_many(predictions_list)
    tdc_mean, tdc_std = [float(value) for value in tdc_raw[name]]

    result = {
        "endpoint": name,
        "metric": "AUPRC",
        "split": "official TDC scaffold split",
        "sota": SOTA,
        "protocol": ("fresh training: 5 independent runs, each = mean of 3 disjoint CatBoost seeds "
                     "(MapLight 2580d + 4x300d frozen Hu2020 GIN embeddings, iter=800, lr=0.1, ss=0.6, cw=sqrt)"),
        "mode": "train",
        "quick": args.quick,
        "runs": [{"run": i, "seeds": g, "auprc": float(average_precision_score(test_y, v))}
                 for i, (g, v) in enumerate(zip(groups, run_vectors), 1)],
        "tdc_evaluate_many": {"mean": tdc_mean, "std": tdc_std},
        "gap_to_sota": tdc_mean - SOTA,
        "beat_sota": tdc_mean > SOTA,
        "fitted_catboost_models": len(args.seeds),
        "dataset_sha256": digest,
        "environment": environment_manifest(),
        "runtime_seconds": time.time() - started,
        "recorded_at": datetime.now().isoformat(),
    }
    result_name = "cyp2d6_inh_smoke_results.json" if args.quick else "cyp2d6_inh_train_results.json"
    prediction_name = "cyp2d6_inh_smoke_predictions.npz" if args.quick else "cyp2d6_inh_train_predictions.npz"
    with open(OUTPUT_DIR / result_name, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(OUTPUT_DIR / prediction_name, y_test=test_y, predictions=np.asarray(run_vectors))

    log("=" * 72)
    log(f"TDC evaluate_many: {tdc_mean:.4f} +/- {tdc_std:.4f}")
    log(f"Gap to SOTA {SOTA:.3f}: {tdc_mean - SOTA:+.4f}")
    log(f"Runtime: {duration(time.time() - started)}")
    log(f"Saved: {OUTPUT_DIR / result_name}")
    log("=" * 72)


# --------------------------------------------------------------------------
# Train-mode internals: MapLight features + frozen GIN embeddings + CatBoost
# --------------------------------------------------------------------------

def _mols(smiles_list):
    from rdkit import Chem
    return [Chem.MolFromSmiles(s) for s in smiles_list]


def _desc2d(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    n = len(smiles_list)
    desc_list = Descriptors._descList
    X = np.zeros((n, len(desc_list)), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is None:
            continue
        for j, (_, func) in enumerate(desc_list):
            try:
                val = func(mol)
                if val is not None and np.isfinite(val):
                    X[i, j] = float(val)
            except Exception:
                pass
    return X


def _morgan_count(smiles_list, radius, bits):
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    n = len(smiles_list)
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=bits)
    X = np.zeros((n, bits), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is not None:
            ConvertToNumpyArray(gen.GetCountFingerprint(mol), X[i])
    return X


def _avalon(smiles_list, bits=1024):
    from rdkit.Avalon import pyAvalonTools
    from rdkit.DataStructs import ConvertToNumpyArray
    n = len(smiles_list)
    X = np.zeros((n, bits), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is not None:
            try:
                ConvertToNumpyArray(pyAvalonTools.GetAvalonCountFP(mol, nBits=bits), X[i])
            except Exception:
                pass
    return X


def _erg(smiles_list):
    from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
    n = len(smiles_list)
    X = np.zeros((n, 315), dtype=np.float32)
    for i, mol in enumerate(_mols(smiles_list)):
        if mol is not None:
            try:
                X[i] = GetErGFingerprint(mol).astype(np.float32)
            except Exception:
                pass
    return X


def maplight_2580d(smiles_list):
    return np.hstack([
        _morgan_count(smiles_list, 2, 1024),
        _avalon(smiles_list, 1024),
        _erg(smiles_list),
        _desc2d(smiles_list),
    ]).astype(np.float32)


def _pure_graph(smiles):
    """Pure-RDKit replica of dgllife smiles_to_bigraph(add_self_loop=True) for
    the PretrainAtom/BondFeaturizer conventions (verified bit-identical against
    dgl on the full test set)."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdmolops, rdmolfiles

    bond_ord = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    bond_dir_ord = [Chem.BondDir.NONE, Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT]
    chir_ord = [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_OTHER]

    mol = Chem.MolFromSmiles(smiles)
    mol = rdmolops.RenumberAtoms(mol, rdmolfiles.CanonicalRankAtoms(mol))
    n = mol.GetNumAtoms()
    atom_id = torch.tensor([a.GetAtomicNum() - 1 for a in mol.GetAtoms()], dtype=torch.long)
    chir = torch.tensor([chir_ord.index(a.GetChiralTag()) for a in mol.GetAtoms()], dtype=torch.long)
    src, dst, bt, bd = [], [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bti = bond_ord.index(b.GetBondType())
        bdi = bond_dir_ord.index(b.GetBondDir())
        src += [u, v]
        dst += [v, u]
        bt += [bti, bti]
        bd += [bdi, bdi]
    src += list(range(n))
    dst += list(range(n))
    bt += [4] * n
    bd += [0] * n
    return {
        "atom_id": atom_id, "chir": chir,
        "bond_type": torch.tensor(bt, dtype=torch.long),
        "bond_dir": torch.tensor(bd, dtype=torch.long),
        "src": torch.tensor(src, dtype=torch.long),
        "dst": torch.tensor(dst, dtype=torch.long),
        "n_nodes": n,
    }


def gin_embeddings(smiles_list, kinds, device, bs):
    import torch
    from gin_pure import GraphBank, mean_pool, state_to_pure

    graphs = [_pure_graph(s) for s in smiles_list]
    bank = GraphBank(graphs, device)
    indices = list(range(len(smiles_list)))
    all_embs = []
    with torch.no_grad():
        for kind in kinds:
            pth = GIN_WEIGHTS_DIR / f"gin_supervised_{kind}_pre_trained.pth"
            if not pth.exists():
                raise FileNotFoundError(f"Missing pretrained GIN weights: {pth}")
            state = torch.load(pth, map_location="cpu", weights_only=False)["model_state_dict"]
            model = state_to_pure(state, device=device)
            embs = []
            for i in range(0, len(indices), bs):
                ids = indices[i:i + bs]
                info = bank.batch(ids)
                h = model([info["atom_id"], info["chir"]], info["src"], info["dst"],
                          [info["bond_type"], info["bond_dir"]])
                embs.append(mean_pool(h, info).cpu().numpy())
            all_embs.append(np.concatenate(embs))
            log(f"    GIN {kind}: {all_embs[-1].shape}")
    return np.hstack(all_embs)


def build_features(train_smiles, test_smiles):
    import torch
    kinds = ["contextpred", "masking", "infomax", "edgepred"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}; extracting frozen GIN embeddings (4 x 300d)...")
    all_smiles = train_smiles + test_smiles
    emb = gin_embeddings(all_smiles, kinds, device, 1024).astype(np.float64)
    emb_tv, emb_te = emb[: len(train_smiles)], emb[len(train_smiles):]
    X_tv = np.hstack([maplight_2580d(train_smiles), emb_tv])
    X_te = np.hstack([maplight_2580d(test_smiles), emb_te])
    return X_tv, X_te


def make_catboost(seed, args):
    import catboost as cb
    kwargs = dict(
        iterations=args.iterations,
        random_strength=2,
        subsample=args.ss,
        sampling_frequency="PerTree",
        loss_function="Logloss",
        auto_class_weights="SqrtBalanced",
        random_seed=seed,
        verbose=0,
        thread_count=args.threads,
        allow_writing_files=False,
    )
    if args.lr > 0:
        kwargs["learning_rate"] = args.lr
    return cb.CatBoostClassifier(**kwargs)


def train_catboost_seed(X_train, y_train, X_test, seed, args, dataset_hash):
    cache = CACHE_DIR / f"cb_seed{seed}_{dataset_hash[:12]}_iter{args.iterations}.npz"
    if args.resume and cache.exists():
        log(f"    CatBoost seed {seed}: cache")
        return np.load(cache)["test"]
    t0 = time.time()
    model = make_catboost(seed, args)
    model.fit(X_train, y_train)
    prediction = model.predict_proba(X_test)[:, 1]
    np.savez_compressed(cache, test=prediction)
    log(f"    CatBoost seed {seed}: {duration(time.time() - t0)}")
    return prediction


def main():
    args = parse_args()
    if args.mode == "reproduce":
        run_reproduce(args)
    else:
        train_mode(args)


if __name__ == "__main__":
    main()


