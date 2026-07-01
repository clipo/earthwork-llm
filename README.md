# EarthworkLLM — label-free LiDAR screening for earthen mounds

Reproducibility code for *"Toward Label-Free Archaeological Prospection: A
Zero-Shot Workflow for Detecting Earthen Mounds from Bare-Earth LiDAR in the
Yazoo Basin"* (Lipo, Davis, and DiNapoli; submitted to *Archaeological
Prospection*). The manuscript is in [`docs/MANUSCRIPT.md`](docs/MANUSCRIPT.md).

The workflow screens public bare-earth LiDAR for compact mound-like features
using **no labeled examples from the target landscape**: a deterministic
multi-scale geomorphon detector, a context-aware False-Positive Shield
(land-cover + mapped modern features + footprint linearity), and a fine-tuned,
zero-shot vision-language interpretation layer whose contribution is measured in
Section 3.6.

## What runs from public data alone

Everything except the vision-language layer runs on the seamless USGS 3DEP
ImageServer with no API key, no local data, and no labeled mounds:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # installs the earthwork_llm package + requirements

# Regional scan of a public site (geomorphon detector + shield), e.g. Jaketown:
python scripts/regional_earthwork_scanner.py \
  --bbox=-90.4949,33.1752,-90.4746,33.1955 \
  --out-dir data/scan_jaketown --tile-size-m 500 --overlap-m 50 --api-url="" --keep-rejected

# Winterville Mound A relief panel (open-terrain positive control):
#   see docs/REPRODUCTION_GUIDE.md
```

See [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md) to reproduce each
manuscript figure and the Table 1 recall curve.

## Data policy — please read

This repository follows a strict [sensitive-site policy](docs/DATA_POLICY.md).
**It contains coordinates only for already-published public monuments**
(Winterville, Lake George/Holly Bluff, Jaketown — `data/reference/published_sites.csv`,
each with a publication citation). The 35-mound reference set used for the
Table 1 recall evaluation contains **restricted site coordinates and is not
distributed here**; qualified researchers can obtain it from the authors and
point the validation scripts at it via the `EARTHWORK_GOLD_LIST` environment
variable. Do not add unpublished site coordinates to this repository.

## The vision-language interpretation layer (optional)

The fine-tuned model (Qwen3-VL-30B-A3B-Thinking + QLoRA adapter, trained
zero-shot on New York State LiDAR) is served separately and is **not** required
for the detection results above. Weights are released via Hugging Face (see the
manuscript's Data and Code Availability); serve with `scripts/serve_yazoo_model.sh`
and `pip install -r requirements-vlm.txt`. Its contribution is measured in Section 3.6: it reads mound-like shape but does
not separate pre-European from modern earthworks.

## Layout

```
src/earthwork_llm/surface/     geomorphons, false-positive shield, DEM/terrain utilities
src/earthwork_llm/ingestion/   3DEP ImageServer fetch, USGS-quad noise-map extraction
scripts/                       scanner, detector, validation (refind_utm), figures, review tools
docs/                          manuscript, reproduction guide, data policy, references
data/reference/                published-site coordinates only
figures/                       manuscript figures
```

## Citation & license

Apache 2.0 (`LICENSE`). If you use this code, please cite the manuscript.
