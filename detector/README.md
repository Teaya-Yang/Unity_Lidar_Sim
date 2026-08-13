# LiDAR 3D object detector

PointPillars (mmdetection3d 1.4.0) replacing the centroid clustering in
`controller/dynamic_clusters.py` as the source of dynamic obstacles.

Chosen over CenterPoint-voxel because the backbone is a 2D CNN over a BEV
pseudo-image — no sparse convolution, so no CUDA requirement. This machine has an
AMD Vega 8 iGPU; ROCm does not support Raven/Picasso, so there is no GPU path.
Training runs off-machine; this machine does data prep, export and inference.

The head regresses `code_size: 9` = `(x, y, z, w, l, h, yaw, vx, vy)`. The yaw and
velocity are what the centroid tracker cannot produce, and what an anisotropic
keep-out needs.

## Environment

```bash
mamba create -y -n lidar3d python=3.10
conda activate lidar3d
export PYTHONNOUSERSITE=1

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install "numpy==1.26.4" "setuptools<70"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
pip install mmdet==3.2.0
pip install --no-build-isolation -e third_party/mmdetection3d
pip install "numba==0.59.1" "opencv-python==4.10.0.84" "plyfile==0.7.4" onnxruntime
```

Pins that are load-bearing:

- `mmcv` from the **prebuilt CPU wheel index**. Building it from source needs
  several GB and OOMs on this machine.
- `numpy<2` — torch 2.1 is binary-incompatible with numpy 2.x. Several transitive
  deps (opencv, numba, plyfile) pull numpy 2 in, hence their pins.
- `setuptools<70` — 84 removed `pkg_resources`, which `torch.utils.cpp_extension`
  imports.
- `--no-build-isolation` — mmdet3d's `setup.py` imports torch at build time.

## Pretrained weights

`load_from` points at the upstream nuScenes checkpoint (mAP 34.33 / NDS 49.1),
downloaded to `checkpoints/`. 126 of 132 tensors load (98.7% of params); the 6 that
clash are the head output convs, which must re-initialise anyway because the class
count differs. Fine-tuning still required — nuScenes is a road dataset with a
different beam pattern.

TensorRT is not an option here: it is CUDA-only, so there is no AMD or CPU backend.
The equivalent path is ONNX Runtime.

## Measured

8-thread Ryzen 5 PRO 3500U, 16200-point scan, uniform random points.

Head cost against `score_thr` — random points are the worst case, since every anchor
fires; a trained model on real scans behaves like the bottom row:

| score_thr | boxes | backbone | head + NMS |
|---|---|---|---|
| 0.30 | 50 | 934 ms | 1912 ms |
| 0.50 | 50 | 947 ms | 922 ms |
| 0.70 | 0 | 936 ms | 75 ms |

So the 2D BEV net dominates realistic cost. Exported to ONNX (`bevnet_*.onnx`,
backbone → neck → head convs):

| | time | size |
|---|---|---|
| torch fp32 | ~940 ms | 19 MB |
| ONNX fp32 | 322 ms | 19 MB |
| ONNX int8 (static QDQ) | 276 ms | 4.8 MB |

INT8 buys little speed because Zen+ has AVX2 but no VNNI; the 4x size reduction is
the real gain on a RAM-starved machine. The int8 model is calibrated on RANDOM
tensors — valid for timing, meaningless for accuracy. Recalibrate on real scatter
outputs before trusting a detection.

## Next

1. Export training data: `scripts/bag_to_nuscenes.py`, then `tools/create_data.py nuscenes`.
   Add `sample_annotation.velocity` (prev/next chains are already written).
2. Fine-tune from `load_from` on a cloud GPU.
3. Re-export and recalibrate ONNX, run in the controller via `onnxruntime` — the ROS
   process should not import torch or mmcv.
