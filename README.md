# ADMETox CYP2D6_Inh TDC

TDC-compliant, fully reproducible submission for the official `CYP2D6_Veith` benchmark. Official-test AUPRC **0.8208 +/- 0.0013**, TDC `evaluate_many` **0.821 +/- 0.001** — beats the TDC leader (MapLight + GNN, `0.790 +/- 0.001`).

## TDC Protocol

- Dataset loader: `tdc.benchmark_group.admet_group`
- Endpoint: `CYP2D6_Veith` (CYP2D6 inhibition)
- Official split: TDC scaffold split, 10,504 `train_val` and 2,626 test molecules
- Metric: AUPRC
- Required evaluation: `group.evaluate_many()` over five independent runs
- Official leaderboard: https://tdcommons.ai/benchmark/admet_group/03cyp2d6_inh/

## Strategy (frozen Big Pickle leg)

```text
prediction(seed) = CatBoost( MapLight 2580d + 4 x 300d frozen Hu2020 GIN embeddings )
run             = mean over 3 disjoint seeds
```

- **MapLight 2580d** (the SOTA feature stack): Morgan r2 count 1024 + Avalon count 1024 + ErG 315 + RDKit2D 217.
- **Frozen GIN embeddings**: 300-d mean-pooled embeddings from the four Hu2020 pretrained GINs (contextpred / masking / infomax / edgepred), weights committed in `assets/gin_weights/`. Purely *frozen* (no fine-tuning) — embeddings are deterministic per molecule.
- **CatBoost**: 800 iterations, lr 0.1, subsample 0.6, `SqrtBalanced` class weights, 15 seeds (1-15).
- The five reported runs are the five disjoint 3-seed groups (1-3, 4-6, 7-9, 10-12, 13-15). All five prediction vectors are distinct.

## Result

Verified run (reproduce mode):

| Run | Seeds | AUPRC |
|-----|-------|-------|
| 1 | 1-3 | 0.8201 |
| 2 | 4-6 | 0.8206 |
| 3 | 7-9 | 0.8233 |
| 4 | 10-12 | 0.8202 |
| 5 | 13-15 | 0.8197 |
| **Mean / std** | | **0.8208 +/- 0.0013** |
| **TDC evaluate_many** | | **0.821 +/- 0.001** |

`group.evaluate_many` returns the mean and std rounded to 3 decimals (`0.821 +/- 0.001`); the unrounded values are `0.8208 +/- 0.0013`.

## Reproduce mode (default)

```bash
python install.py
python run_cyp2d6_inh.py
```

`run_cyp2d6_inh.py` defaults to `--mode reproduce`, which builds the five runs deterministically from the committed per-seed predictions in `assets/cyp2d6_inh_legs.npz` (15 CatBoost seeds), computes each run AUPRC, checks that all five vectors are distinct, and calls `group.evaluate_many`. No models are trained, no randomness is used, so a fresh clone reproduces `output/cyp2d6_inh_results.json` field-for-field.

## Train mode (full end-to-end training)

```bash
python run_cyp2d6_inh.py --mode train
```

This retrains every model from scratch on the downloaded official benchmark: MapLight 2580d features, frozen GIN embeddings from the committed pretrained weights, then the 15 CatBoost seeds, grouped into five independent runs and scored with `evaluate_many`. CatBoost itself is fully deterministic for identical input features, and the frozen embeddings are the same as in the committed leg; only GPU float rounding (~1e-7) in embedding extraction can shift a CatBoost split, so a fresh train run reproduces the strategy and the reported score distribution (typically within +/-0.001 per seed), not the exact committed digits. Training cache under `output/cache/` resumes interrupted runs; use `--no-resume` to retrain everything. `--quick` runs a short smoke variant that writes separate smoke outputs.

## Exact Reproduction

Python 3.12.13 is the verified environment.

### Windows

```powershell
git clone https://github.com/Recconnect/admetox-cyp2d6-inh-tdc.git
Set-Location admetox-cyp2d6-inh-tdc
py -3.12 -m venv .venv
.venv\Scripts\python.exe install.py
.venv\Scripts\python.exe -u run_cyp2d6_inh.py
```

### Linux

```bash
git clone https://github.com/Recconnect/admetox-cyp2d6-inh-tdc.git
cd admetox-cyp2d6-inh-tdc
python3.12 -m venv .venv
.venv/bin/python install.py
.venv/bin/python -u run_cyp2d6_inh.py
```

`install.py` installs the pinned `requirements.txt`, then installs PyTDC with `--no-deps` (its optional `cellxgene-census` dependency is incompatible with Python 3.12). The first run downloads the official benchmark into `data/`.

## Outputs

- `output/cyp2d6_inh_results.json`: five run scores, precise mean/std, TDC `evaluate_many`, exact seeds, dataset hash, environment and runtime.
- `output/cyp2d6_inh_predictions.npz`: five distinct official-test prediction vectors.
- `assets/cyp2d6_inh_legs.npz`: committed per-seed predictions used by reproduce mode (15 CatBoost seeds).
- `assets/gin_weights/`: the four Hu2020 pretrained GIN checkpoints used by train mode.
- `output/cache/`: resumable seed-level training predictions, ignored by Git.

## Hardware

- AMD Radeon RX 6900 XT, 16 GB VRAM
- ROCm 7.12 compatible PyTorch 2.10.0
- AMD Ryzen 9 3900X
- Windows 11

Embedding extraction runs on the available CUDA/ROCm device; CPU execution is supported but slower. CatBoost runs on CPU. Reproduce mode runs on any device and does not train.

## TDC Submission

Submission-ready values and metadata are recorded in `SUBMISSION.md`. TDC submission instructions: https://tdcommons.ai/benchmark/overview/

## License

MIT License. See `LICENSE`.
