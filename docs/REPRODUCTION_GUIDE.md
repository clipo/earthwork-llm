# Reproduction guide

How to reproduce the manuscript's results from public data. Everything below,
except the Table 1 recall evaluation (which needs the restricted reference set,
see [DATA_POLICY.md](DATA_POLICY.md)), runs on the seamless USGS 3DEP ImageServer
with no API key and no local data.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # earthwork_llm + requirements.txt
```

`pdal` is needed only if you re-ground-filter raw point clouds; the ImageServer
path used throughout needs only the pip requirements.

## §3.3 — Jaketown shield case study (Figure 7)

```bash
python scripts/regional_earthwork_scanner.py \
  --bbox=-90.4949,33.1752,-90.4746,33.1955 \
  --out-dir data/scan_jaketown --tile-size-m 500 --overlap-m 50 \
  --api-url="" --keep-rejected
```

Produces `regional_detections.{csv,geojson}`; the shield funnel (277 raw →
197 rejected → 80 survivors) is tabulated from the `shield_decision` column.
`scripts/build_review.py data/scan_jaketown` builds relief thumbnails for the
survivor review UI (`scripts/review_server.py`).

## §3.4 — Lake George canopy negative & Winterville positive control (Figs 8, 9)

Re-run the scan at the true Lake George Mound A site (32.785, −90.785) and at
Winterville Mound A (see `data/reference/published_sites.csv`); the Lake George
AOI returns a near-flat surface (canopy), while Winterville plainly resolves the
~12 m platform. `scripts/build_correct.py` renders the relief panels.

## §3.5 — Agricultural-island shield behaviour (Figure 10)

Run the scanner over a Cultivated-Crops tract; compact rises are kept as islands
while linear plow/road/canal features are rejected. The positive-relief +
compactness gate discussed in §3.2 is a two-line filter on the survivor
`height_m` and `area_m2` fields.

## §3.1 — Recall vs. tolerance, Table 1 (RESTRICTED reference set)

Requires the 35-mound reference set, which is **not** distributed (DATA_POLICY.md):

```bash
export EARTHWORK_GOLD_LIST=/path/to/located_mounds.csv
python scripts/refind_utm.py       # recall at 10/15/20/25/30 m in UTM 15N + offsets
python scripts/gen_fig6_utm.py     # regenerates Figure 6 from the UTM run
```

`refind_utm.py` fetches a 300 m tile centred on each reference point in UTM
Zone 15N (true metres), runs the single-scale geomorphon detector
(`classify_geomorphon_simple`, 5 m radius, 0.3 m flatness) plus
`detect_earthworks`, and scores a hit if any of the detector's ten candidates
falls within the tolerance.

## The vision-language layer (optional)

Not required for any result above. `pip install -r requirements-vlm.txt`, then
`scripts/serve_yazoo_model.sh` on a CUDA host to serve the V9.1 adapter (weights
via Hugging Face; see the manuscript). Passing `--api-url` to the scanner then
sends shield survivors to the model. Its contribution to detection is not yet
measured; the manuscript identifies the ablation as future work.

## Notes on coordinate systems

Table 1 (recall) is computed in UTM Zone 15N (EPSG:26915) so distances are true
metres. The regional scanner runs in Web Mercator (EPSG:3857); at ~33° N its
distances are ~19% and areas ~42% larger than true ground (see manuscript §2.1).
