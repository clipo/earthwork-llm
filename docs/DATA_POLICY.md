# Data policy — sensitive site coordinates

This project locates pre-European earthen mounds. Precise coordinates of
archaeological sites are sensitive: publishing them can enable looting and
damage. This repository is therefore governed by a strict rule.

## The rule

**Only coordinates of already-published public monuments are committed here.**
That set is `data/reference/published_sites.csv` — Winterville Mound A
(Winterville Mounds State Park), Lake George / Holly Bluff (Williams & Brain
1983), and the Jaketown complex (a National Historic Landmark) — each carrying a
publication or public-status citation.

## What is deliberately **not** in this repository

- The 35-mound reference set used for the Table 1 recall evaluation
  (`located_mounds.csv`), which draws on Lower Mississippi Survey grid-quadrat
  coordinates of restricted precision.
- The Eskew ground-survey eval set (`mounds_seed.csv`) used for the Section 3.6
  ablation, which carries site coordinates. The coordinate-free verdict results
  (`data/vlm_ablation/ablation_results.csv`) are shipped; the coordinates are not.
- Any per-scan detection output that reports coordinates near unpublished sites.
- The South Carolina shell-ring gold list (`SC_Gold_List.csv`, Davis 2018/2019) used for the Section 3.7 generalization test, which carries site coordinates. Provide it via `SHELL_RING_GOLD`.
- Any state site-file coordinates.

`.gitignore` blocks `located_mounds.csv`, `*goldlist*.csv`, `validation_summary.csv`,
and all of `data/` except the published-sites file, as a backstop.

## Reproducing the restricted recall evaluation

Qualified researchers can obtain the 35-mound reference set from the authors and
run:

```bash
export EARTHWORK_GOLD_LIST=/path/to/located_mounds.csv   # not distributed here
python scripts/refind_utm.py       # Table 1 recall-vs-tolerance, UTM 15N
python scripts/gen_fig6_utm.py     # Figure 7

export EARTHWORK_ABLATION_SET=/path/to/mounds_seed.csv   # not distributed here
python scripts/vlm_ablation.py     # Section 3.6 ablation (also needs a served model)
```

## Contributing

Do not open pull requests that add unpublished site coordinates, or detection
outputs that expose them. Contributions that would weaken this policy will be
declined.
