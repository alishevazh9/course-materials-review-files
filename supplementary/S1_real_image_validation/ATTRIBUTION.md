# Supplementary File S1 — technical validation on real underwater images

Single-class fish detection with YOLOv8n fine-tuned on a Fish4Knowledge-derived dataset
(manuscript Sections 3.10 and 4.5).

## Contents

| Path | Content |
|---|---|
| `Fish4Knowledge_YOLOv8n_oneclass.ipynb` | Google Colab notebook: download, merge of numeric categories into one class "fish", training, test evaluation |
| `yolov8n_fish_oneclass/args.yaml` | Full training configuration (30 epochs, 640 × 640, batch 16, seed 42, deterministic) |
| `yolov8n_fish_oneclass/experiment_metadata.txt` | Dataset, split sizes, GPU, training time |
| `yolov8n_fish_oneclass/results.csv`, `results.png` | Per-epoch losses and validation metrics (final epoch: P 0.881, R 0.851, mAP@50 0.922, mAP@50–95 0.582) |
| `yolov8n_fish_oneclass/weights/best.pt` | Trained weights (Ultralytics 8.4.155, version recorded in the checkpoint) |
| `yolov8n_fish_oneclass/*.png, *.jpg` | Training curves, confusion matrices, training and validation batches |
| `test_oneclass/test_metrics.txt` | Held-out test metrics: 188 images, 751 fish; P 0.887, R 0.885, mAP@50 0.934, mAP@50–95 0.576 |
| `test_oneclass/*` | Test-split precision–recall and F1 curves, confusion matrices, example detections (labels vs. predictions) |

Paths such as `/content/drive/MyDrive/...` in `args.yaml` and in the notebook are generic Google Colab
paths and contain no personal information.

## Third-party data and attribution

The source images are **not** redistributed here; the dataset can be downloaded from its original
location. The image mosaics in this folder (`train_batch*.jpg`, `val_batch*.jpg`) are derived from
that dataset and are shared under its CC BY 4.0 licence.

* Dataset: g18L5754 (2023). *Fish4Knowledge dataset* (Version 1). Roboflow Universe.
  https://universe.roboflow.com/g18l5754/fish4knowledge-dataset/dataset/1 — licence CC BY 4.0.
* Original source: Fisher, R. B., Chen-Burger, Y.-H., Giordano, D., Hardman, L., & Lin, F.-P. (Eds.).
  (2016). *Fish4Knowledge: Collecting and analyzing massive coral reef fish video data*. Springer.
  https://doi.org/10.1007/978-3-319-30208-9
* Detector: Ultralytics YOLOv8 (Jocher, Chaurasia & Qiu, 2023), AGPL-3.0; COCO-pre-trained weights.

## Limits

Single-class detection only (not species recognition). Neighbouring video frames may be near-duplicates,
so a stricter protocol would split frames by source video. These results show that the course pipeline
runs on real imagery; they are not evidence of learning gains or farm readiness.
