# C1 scorer — standalone AP / reach for Grounded-SAM recording banks

Runs the deployed Component-1 backend over recording banks and scores them with the untouched
official class-agnostic ScanNet evaluator. Self-contained: no dependency on the SpaceSculptor
checkouts, all data paths supplied by flag or environment.

## What it reports

| column | meaning |
|---|---|
| `masks/sc` | recorded 2D masks lifted into the scene — a front-end property |
| `reach@.25` | share of eligible GT that ANY single recorded mask already covers at IoU .25. An upper bound no backend can exceed, and the stable signal on small samples |
| `AP / AP50 / AP25` | official class-agnostic AP. Set-level, so it needs a sizeable scene count before it means anything |
| `rc@.25` | the evaluator's own recall |
| `preds` | predictions surviving the evaluator's 100-vertex floor |

## Protocol

Per scene: relift → retention absorb → residual discovery → anchored variation-of-information
joint inference at hypothesis-bank floor **0.55**. Every prediction is frozen before ground truth
is opened. Scored at **constant export confidence 1.0** — the convention CDIS and MV3DIS use.

With several banks it restricts to the scenes they all contain, which is what makes the comparison
matched. It scores whatever exists, so it can be pointed at a run still in progress.

## Install

```bash
pip install numpy scipy torch plyfile scikit-learn
```

No GPU needed — the backend is CPU (numpy/scipy). Recording needs a GPU; scoring does not.

## Data required

| | |
|---|---|
| `--scans` | ScanNet scans dir: per scene `<scene>_vh_clean_2.ply`, `<scene>_vh_clean_2.0.010000.segs.json`, `<scene>.aggregation.json`, `<scene>.txt`. **No `.sens` needed** — scoring never reads RGB-D |
| `--label-map` | `scannetv2-labels.combined.tsv` |
| `--s200` | ScanNet200 `val/` `.pth` labels — only for `--annotation scannet200` |

Scoring needs about **9 MB per scene**, not the 700 MB the recorder needs.

## Run

```bash
python score.py \
  --banks ours=/data/banks/v5_gdsam_v96vocab open3dis=/data/banks/gdsam_o3d \
  --scans      /data/scannet/scans \
  --label-map  /data/scannet/scannetv2-labels.combined.tsv \
  --s200       /data/scannet200/val \
  --annotation scannetv2 --workers 12
```

`--annotation scannet200` scores the 198-category re-annotation of the same scans.
`--limit N` caps the scene count for a quick check.

Paths may also be given as `C1_SCANS`, `C1_LABEL_MAP`, `C1_S200`.

## Example

```
  2 scene(s) in all 2 bank(s), scannetv2, bank floor 0.55, constant confidence 1.0

  bank                          masks/sc  reach@.25       AP     AP50     AP25   rc@.25   preds
  chunk1                            1248      96.2%   50.613   85.714   91.003   96.154      72
  chunk10                           1068      96.2%   49.728   80.563   91.003   96.154      68
```

Read `reach@.25` on small samples and `AP*` only once the scene count is substantial: AP is a
set-level quantity, and with a few dozen GT instances one instance crossing a threshold moves it
by several points.
