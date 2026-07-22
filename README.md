# EarthworkLLM — label-free LiDAR screening for earthen mounds

Reproducibility repository for *"Toward Zero-Shot Archaeological Prospection:
Training-Free Detection of Pre-European Earthworks from Public LiDAR in the
American Southeast"* (Lipo, Davis, and DiNapoli; submitted to *Archaeological
Prospection*). The manuscript is [`docs/MANUSCRIPT.md`](docs/MANUSCRIPT.md);
every reported number maps to a script and released output in
[`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md) and the manuscript's
Table 3.

The workflow screens public bare-earth LiDAR for earthen features using **no
labeled examples from the target landscape**. A geomorphon transform turns the
landscape into a text in a ten-class landform vocabulary, a deterministic
detector extracts candidate earthworks, a rule-based False-Positive Shield
screens them against land-use context, and computed relief plus a
vision-language model's structured reading rank the survivors for triage
(manuscript Sections 2 to 4).

## What reproduces from what

| Tier | What | Needs |
|---|---|---|
| 1 | Regional scans, shield behavior, two-score triage columns, case-study and explanatory figures | this repo + internet only (public USGS 3DEP ImageServer and public land-use services; no API keys, no local data) |
| 2 | Recall/decoy/relief evaluations (Table 1, Figure 6, Appendix B.1) | Tier 1 + the restricted 35-mound reference set via `EARTHWORK_GOLD_LIST` (see Data policy) |
| 3 | Vision-language arms (Sections 2.5, 3.6, 3.8; Appendices B.3, B.6) | Tier 1 or 2 + the released model weights served locally (hardware below); desk-review labels via `JAKETOWN_VERDICTS` for the Section 3.8 ranking |

All model-arm *outputs* are released in `data/v10_eval/`, so the Tier 3
analyses and Figure 13a re-derive from the shipped CSVs without any GPU.

## Hardware requirements

- **Tier 1 and 2 (detector, shield, evaluations, most figures):** any 64-bit
  machine with 8 GB RAM and internet access. No GPU. A full county-scale scan
  is network-bound; the Jaketown quickstart below takes a few minutes.
- **Tier 3 (serving the vision-language model):** a CUDA GPU (or unified-memory
  system) with at least **64 GB of GPU-addressable memory** for the bf16 model
  (Qwen3-VL-30B-A3B-Thinking, ~61 GB of weights) plus ~80 GB free disk. We used
  a single NVIDIA GB10 (128 GB unified memory). FP8 on-the-fly quantization
  (`--quantization fp8`) fits in ~40 GB with some numeric drift from the
  reported runs. Serving uses vLLM, which requires Linux + CUDA.

## Setup

Python 3.10 or newer.

**Linux / macOS**

```bash
git clone https://github.com/clipo/earthwork-llm.git
cd earthwork-llm
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Windows** — use WSL2 (Ubuntu) and follow the Linux steps inside WSL. Native
Windows works for Tiers 1 and 2 (`py -m venv .venv` then
`.venv\Scripts\activate`), but the geospatial wheels and vLLM are far better
supported under WSL2, and Tier 3 serving requires Linux + CUDA regardless.

**macOS note** — Tiers 1 and 2 run natively on Apple Silicon or Intel. Tier 3
serving does not (vLLM needs CUDA); either serve the model on a Linux GPU host
and point `VLM_API` at it, or work from the released CSVs.

Figure 1 additionally needs `cartopy` (`pip install cartopy`).

## Quickstart (verify your setup, ~5 minutes, Tier 1)

```bash
# Regional scan of the public Jaketown site (geomorphon detector + shield).
# Output CSV includes the two-score triage columns of manuscript Sections 3.8/4:
# score_a (artificial vs natural) and score_b (modern association), with ranks.
python scripts/regional_earthwork_scanner.py \
  --bbox=-90.4949,33.1752,-90.4746,33.1955 \
  --out-dir data/scan_jaketown --tile-size-m 500 --overlap-m 50 --api-url="" --keep-rejected
```

You should see ~277 raw candidates with the shield rejecting ~71% (manuscript
Section 3.3). Then follow `docs/REPRODUCTION_GUIDE.md` result by result. It
gives the exact command, inputs, and released output for every reported number,
and a figure-by-figure map for every manuscript figure.

## Data policy — please read

This repository follows a strict [sensitive-site policy](docs/DATA_POLICY.md).
**It contains coordinates only for already-published public monuments**
(Winterville, Lake George/Holly Bluff, Jaketown — `data/reference/published_sites.csv`,
each with a publication citation). The 35-mound reference set used for the
Table 1 recall evaluation contains restricted site coordinates and is **not
distributed**. Qualified researchers can obtain it from the corresponding
author (clipo@binghamton.edu) and point the validation scripts at it via the
`EARTHWORK_GOLD_LIST` environment variable; the Jaketown desk-review file
(`JAKETOWN_VERDICTS`) is available under the same terms. Do not add unpublished
site coordinates to this repository.

## The vision-language layer

The reader is Qwen3-VL-30B-A3B-Thinking with a QLoRA adapter trained zero-shot
on New York State LiDAR (manuscript Section 2.5.3). Weights are released via
Hugging Face, linked from the manuscript's Data and Code Availability. Serve
with:

```bash
pip install -r requirements-vlm.txt
bash scripts/serve_yazoo_model.sh        # vLLM, Linux + CUDA
```

Its measured roles: it reads earthwork form zero-shot but does not separate
pre-European from modern earthworks (Section 3.6), model-only localization
performs at its decoy floor (Section 3.1), and its feature-scale isolation
reading fuses with computed relief into the significant triage ranking of
Section 3.8. All arm outputs are released in `data/v10_eval/`, and the
statistical analyses reproduce from the CSVs alone.

## Layout

```
src/earthwork_llm/surface/     geomorphons, false-positive shield, triage scores, DEM utilities
src/earthwork_llm/ingestion/   3DEP ImageServer fetch, USGS-quad noise-map extraction
scripts/                       scanner, detectors, evaluation arms, context sheets, figure generators
docs/                          manuscript, reproduction guide (with figure map), data policy
data/reference/                published-site coordinates only
data/v10_eval/  data/refind_utm/  ...   released outputs behind every reported number
docs/figures/                  manuscript figures
```

## Citation & license

Apache 2.0 (`LICENSE`). If you use this code, please cite the manuscript.
Corresponding author: Carl P. Lipo, Binghamton University (clipo@binghamton.edu).
