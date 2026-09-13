# Grounded-SAM front end, Open3DIS configuration — multi-machine recorder

Reproduces the 2D front end Open3DIS ships and MV3DIS compares against, over all 312 ScanNet
validation scenes, split across as many machines and GPUs as you have.

The configuration is read from Open3DIS's own `configs/scannet200.yaml` and `tools/text_query.py`,
not assumed:

| | |
|---|---|
| detector | GroundingDINO **Swin-T** (`groundingdino_swint_ogc.pth`) |
| segmenter | SAM **ViT-H** (`sam_vit_h_4b8939.pth`) |
| thresholds | box 0.4, text 0.4 |
| prompt | the 198 ScanNet200 instance categories |
| query protocol | `--chunk 1` = **one class per query** (`segment_size = 1`) |
| filters | box NMS 0.5, then drop boxes covering >85% of the frame |
| views | stride 10, 200 frames/scene |

## On each machine

```bash
bash setup.sh                        # ~15 min; venv + deps + CUDA op + 3.1 GB of checkpoints
bash setup.sh /path/to/python        # or reuse an existing interpreter, e.g. a conda env
```

With no argument it builds a **venv** at `./venv` and installs everything there, including torch
(cu128 wheels, which cover A6000 sm_86 / 4090 sm_89 / 5090 sm_120). This is deliberate: Debian and
Ubuntu mark the system Python as externally managed (PEP 668) and refuse pip installs with
`error: externally-managed-environment`. Passing `--break-system-packages` can damage the OS
python, so this sidesteps it. If you pass an interpreter that *can* install into itself (a conda
env), it is used as-is.

If venv creation fails, install it first:
```bash
sudo apt install -y python3-venv python3-full
```

`setup.sh` also patches GroundingDINO's CUDA source before building it. Upstream still calls
`AT_DISPATCH_FLOATING_TYPES(value.type(), ...)`, and torch ≥ 2.4 dropped the implicit
`DeprecatedTypeProperties → c10::ScalarType` conversion that relies on, so the build fails with

```
error: no suitable conversion function from "const at::DeprecatedTypeProperties"
       to "c10::ScalarType" exists
```

Two lines are affected (forward and backward); the patch rewrites them to `value.scalar_type()`
and is idempotent. The remaining `.type().is_cuda()` calls are deprecation *warnings* and compile
fine, so they are left alone.

The resolved interpreter is written to `.python_path`, and `run_node.sh` picks it up automatically —
no need to pass it again.

`setup.sh` is idempotent. It pins `transformers==4.44.2` deliberately — v5 removed
`BertModel.get_extended_attention_mask`, which GroundingDINO calls directly, and the failure is an
obscure `AttributeError`. It also leaves `TORCH_CUDA_ARCH_LIST` unset so the MultiScaleDeformableAttn
op autodetects the local card; a prebuilt `_C.so` from another python version or arch will NOT load.

### ScanNet scans

Each machine needs the scans for the shards it owns. `fetch_scannet.sh` pulls only the five file
types used here, only for the shards you ask for:

```bash
bash fetch_scannet.sh ./scans                 # all 312 scenes   (~95 GB)
bash fetch_scannet.sh ./scans 2,3,4,5 6       # just those shards (~63 GB)
bash fetch_scannet.sh ./scans 1 6             # one shard         (~16 GB)
```

It is resumable and idempotent — re-run until it reports 0 incomplete. Two details it handles that
the bundled `download-scannet.py` does not: the `.sens` stream is served from `v1/scans` (the `v2`
path 404s for it, while the mesh and `.txt` come from `v2`), and a single stream runs at roughly
150 KB/s, so scans are fetched concurrently (`PARALLEL`, default 8) because the transfer is
latency-bound rather than bandwidth-bound.

**ScanNet is released under its own Terms of Use — by downloading you agree to them. See
<http://www.scan-net.org/>.** The script retrieves the same files the official toolkit does and
grants no additional rights.

## Then split the work

`TOTAL_SHARDS` is a constant shared by the whole fleet. Give each machine the shard indices it
owns — **more shards than GPUs**, allocated in proportion to each card's speed, so every machine
finishes at about the same time instead of the fleet waiting on the slowest box. A GPU handed
several shards works through them in sequence; only one worker runs per GPU at a time, because a
single worker already saturates a modern card at this batch size.

Example fleet — one 5090, one A6000, one laptop 4090 — split 5 : 3 : 2 over 10 shards:

```bash
# 5090 box            157 scenes
bash run_node.sh ./scans ./out 0,1,2,3,4 10

# A6000 box            93 scenes
bash run_node.sh ./scans ./out 5,6,7     10

# 4090 laptop box      62 scenes, smaller batch for 16 GB
bash run_node.sh ./scans ./out 8,9       10   "" 1 6
```

The 6th and 7th arguments are `CHUNK` and `GD_BATCH`; pass `""` for the interpreter to keep the one
`setup.sh` recorded. **A 16 GB laptop GPU needs `GD_BATCH 6`** — the default of 10 uses about 9.5 GB
for activations on top of 3.3 GB of weights.

Fetch only what each box will process, using the same shard arguments:

```bash
bash fetch_scannet.sh ./scans 0,1,2,3,4 10     # ~47 GB
bash fetch_scannet.sh ./scans 5,6,7     10     # ~28 GB
bash fetch_scannet.sh ./scans 8,9       10     # ~19 GB
```

Shards are round-robin over the scene list, so each gets a fair mix of large and small scenes.
Every scene is written atomically (`.tmp` then rename) and existing files are skipped, so runs are
restartable and outputs merge by plain copy:

```bash
rsync -a a6000box:~/gdsam-open3dis-recorder/out/ ./out/
rsync -a laptop:~/gdsam-open3dis-recorder/out/   ./out/
ls out/*.pkl | wc -l      # 312 when complete
```

## Cost

Measured on a 5090: ~26 image-caption pairs/s at `--gd-batch 10`.

Throughput scales with the fleet's combined rate. Rough per-card rates at `--gd-batch 10`:
5090 ~26 pairs/s (measured), A6000 ~13, laptop 4090 ~10 — about **49 pairs/s** for the three
together.

| | pairs | 5090 alone | 5090 + A6000 + 4090 laptop |
|---|---:|---:|---:|
| `--chunk 1` (Open3DIS exactly) | 12.4 M | ~130 h | **~70 h** |
| `--chunk 10` | 1.25 M | ~13 h | **~7 h** |

`--chunk` is the one knob that trades fidelity for time. Measured on 8 frames of scene0011_00:
chunk 1 → 65 masks, chunk 10 → 50, chunk 99 → 31. The 198-class caption is 529 BERT tokens against
the model's 256-token limit, which is why Open3DIS chunks at all; 99 classes (237 tokens) is the
largest that still fits. **Use `--chunk 1` for the claim and `--chunk 10` for a fast indicative
run.**

## Output

One pickle per scene with `V`, `Vis`, `P`, `mask_frame`, `mask_concept`, `concepts`, `nF`, `nM` —
the recording format the existing backend reads unchanged — plus full provenance (checkpoints,
thresholds, NMS, chunk size, view sampling), which the legacy GD-SAM bank does not record.
