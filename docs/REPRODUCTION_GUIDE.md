# Reproduction guide

How to reproduce the manuscript's results from public data. Everything below,
except the Table 1 recall evaluation and the Section 3.6 ablation (which need
restricted data, see [DATA_POLICY.md](DATA_POLICY.md)), runs on the seamless USGS 3DEP ImageServer
with no API key and no local data.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # earthwork_llm + requirements.txt (the full core path)
```

The core path — 3DEP ImageServer fetch, geomorphon detector, False Positive
Shield, and the Section 3.1 validation — needs only the base install; it was
verified end to end from a clean environment. Optional extras enable optional
inputs: `pip install -e .[quads]` (plus `poppler` and a `conda`-installed PDAL/GDAL)
to build the USGS-quad noise map; `.[gcs]` to write to a cloud bucket; `.[vlm]`
to serve the interpretation layer. `pdal` is needed only to re-ground-filter raw
point clouds; the ImageServer path does not use it.

The 3DEP ImageServer occasionally returns a transient 502 for a tile; the
scanner logs it and continues, and re-running fills any gap.

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

The reported funnel used a modern-feature **noise map** so the shield could
reject canal- and levee-margin candidates from USGS quads. Build it (optional)
with `pip install -e .[quads]` and:

```bash
python scripts/generate_yazoo_noise_map.py --bbox=-90.4949,33.1752,-90.4746,33.1955 \
  --out data/jaketown_noise.geojson
# then pass it to the scanner:  --noise-map data/jaketown_noise.geojson
```

Without `--noise-map` the scanner still runs and logs that USGS/HTMC screening is
inactive; the shield then relies on NLCD land cover and footprint linearity
alone, and the exact funnel counts differ.

## §3.4 — Lake George canopy negative & Winterville positive control (Figs 8, 9)

Re-run the scan at the true Lake George Mound A site (32.785, −90.785) and at
Winterville Mound A (see `data/reference/published_sites.csv`); the Lake George
AOI returns a near-flat surface (canopy), while Winterville plainly resolves the
~12 m platform. `scripts/build_correct.py` renders the relief panels.

## §3.5 — Agricultural-island shield behavior (Figure 10)

Run the scanner over a Cultivated-Crops tract; compact rises are kept as islands
while linear plow/road/canal features are rejected. The positive-relief +
compactness gate discussed in §3.2 is a two-line filter on the survivor
`height_m` and `area_m2` fields.

## §3.1 — Recall vs. tolerance, Table 1 (RESTRICTED reference set)

Requires the 35-mound reference set, which is **not** distributed (DATA_POLICY.md):

```bash
export EARTHWORK_GOLD_LIST=/path/to/located_mounds.csv
python scripts/refind_utm.py       # recall at 10/15/20/25/30 m in UTM 15N + offsets
python scripts/gen_fig6_utm.py     # regenerates Figure 6 (one panel per distinct site) from the UTM run
```

`refind_utm.py` fetches a 300 m tile centered on each reference point in UTM
Zone 15N (true meters), runs the single-scale geomorphon detector
(`classify_geomorphon_simple`, 5 m radius, 0.3 m flatness) plus
`detect_earthworks`, and scores a hit if any of the detector's ten candidates
falls within the tolerance.

## §3.6 — Interpretation-layer ablation, Table 2 (RESTRICTED eval set + served model)

Tests whether the fine-tuned model separates real mounds from modern earthworks
on Eskew's field-verified set. Needs a served model (see below) and the eval set,
which carries coordinates and is **not** distributed (DATA_POLICY.md). The
coordinate-free verdict results are provided at
`data/vlm_ablation/ablation_results.csv` and `ablation_summary.json`.

```bash
export EARTHWORK_ABLATION_SET=/path/to/mounds_seed.csv
export VLM_API=http://localhost:8000/v1/chat/completions
export VLM_MODEL=terrallm-v91
python scripts/vlm_ablation.py            # 3 runs per site, majority vote
```

`vlm_ablation.py` fetches a ~160 m UTM tile per site, builds the same six-panel
image the scanner sends the model, asks for a MOUND / NOT_MOUND verdict, and
scores the majority vote against field truth (confusion matrix, recall,
modern-earthwork rejection). The shipped run gives 4 of 6 mounds kept and only
3 of 22 modern earthworks rejected: the model reads mound-like shape but does
not filter for age, which is the shield's job (manuscript Section 3.6).

## §3.7 — Shell-ring generalization test (RESTRICTED gold list)

Tests generalization to Davis's coastal South Carolina shell rings (a different
feature type and region) on the same public 3DEP product, no retraining. Needs the
SC gold list, which carries site coordinates and is **not** shipped (DATA_POLICY.md).

```bash
export SHELL_RING_GOLD=/path/to/SC_Gold_List.csv
python scripts/shell_ring_test.py   # UTM 17N; per-sector oval/arc-tolerant ring detector
```

Writes `shell_ring_results.csv` and `shell_ring_gallery.png`. The catalogue point
falls within a detected ring footprint for 7 of 10 sites (manuscript Section 3.7,
Figure 11); a production ring detector and transferable discrimination are future work.
`generate_shell_ring_fig.py` rebuilds the Figure 11 map/signature/gallery panel
(needs `pip install cartopy` for the location map).

## Appendix B companions: baseline, shield test, negative control

```bash
# B.1 baseline: LRM blob detector under the identical recall protocol
export EARTHWORK_GOLD_LIST=/path/to/located_mounds.csv   # restricted
python scripts/lrm_baseline.py            # 22/35 at 30 m vs geomorphons' 31/35

# B.3 shield discrimination on the Eskew set (NLCD + linearity; proximity inactive)
export EARTHWORK_ABLATION_SET=/path/to/mounds_seed.csv   # restricted
python scripts/shield_eskew_test.py       # mounds 6/6 kept; modern earthworks 2/22 rejected

# §3.7 negative control: frozen ring detector at decoy points 500 m off-site
export SHELL_RING_GOLD=/path/to/SC_Gold_List.csv         # restricted
python scripts/ring_negative_control.py   # decoys score ~52%/65%, i.e. chance

# §3.6 arm B: context-conditioned ablation (adds a 600 m wide view; needs served model)
python scripts/vlm_ablation_context.py
```

## The vision-language layer (optional)

Required only for the Section 3.6 ablation above. `pip install -r requirements-vlm.txt`, then
`scripts/serve_yazoo_model.sh` on a CUDA host to serve the V9.1 adapter (weights
via Hugging Face; see the manuscript). Passing `--api-url` to the scanner then
sends shield survivors to the model. Its contribution is measured in Section 3.6: it reads mound-like shape but does
not separate pre-European from modern earthworks (that is the shield's job).

## Notes on coordinate systems

Table 1 (recall) is computed in UTM Zone 15N (EPSG:26915) so distances are true
meters. The regional scanner runs in Web Mercator (EPSG:3857); at ~33° N its
distances are ~19% and areas ~42% larger than true ground (see manuscript §2.1).
