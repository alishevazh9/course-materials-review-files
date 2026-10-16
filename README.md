# Reproducible course materials — AI and machine learning for aquaculture undergraduates

Anonymised supplementary package for peer review. It contains everything referred to in the manuscript's
*Data and materials availability* statement: the 15 practical-work notebooks, the synthetic teaching
datasets, the assessment instruments, the build script that regenerates the data, and Supplementary
Files S1 and S2. Version **1.0.3** (see `CHANGELOG.md`).

## Quick check for reviewers (about 1 minute, CPU only)

```bash
pip install -r requirements.txt          # Python 3.10–3.13
python supplementary/S2_verification/verify_reported_values.py
```

The script recomputes 62 values reported in Table 5, Section 4.2 and Table 6 directly from `data/` and
prints PASS / FAIL for each. Expected result with the pinned versions: 61 PASS, 1 version-sensitive
(ARIMA, see below), 0 FAIL. Saved outputs of this script are in `supplementary/S2_verification/`.

To see the notebooks with their outputs without running anything, open
`supplementary/S2_verification/executed_notebooks/`.

## Contents

| Path | What it is | Manuscript |
|---|---|---|
| `notebooks/module_1…5/` | 15 practical works (PW01–PW15), three per module; teaching language Russian | Table 3, Figure 2 |
| `data/` | Synthetic teaching datasets (seed 2607, CC0) and the synthetic fry images with YOLO labels | Table 5 |
| `assessments/rubric.md` | Analytic rubric, 4 criteria × 0–4 points | Table A1 |
| `assessments/pre_post_blueprint.csv` | 20-item blueprint of the parallel pre/post forms | Table A2 |
| `docs/article_alignment.csv` | Mapping of each practical work to its module, evidence and claim limit | Table 12 |
| `build_repository.py` | Regenerates all datasets, notebooks and assessment files deterministically | Section 3.7 |
| `supplementary/S1_real_image_validation/` | YOLOv8n run on real underwater images: notebook, configuration, per-epoch results, test metrics, curves, example detections, weights | Sections 3.10, 4.5 |
| `supplementary/S2_verification/` | Execution log of all notebooks, executed copies, value-check script and its outputs | Sections 3.9, 4.4, Table 8 |
| `requirements.txt` | Pinned CPU dependencies for PW01–PW15 | Section 3.9 |
| `requirements-gpu.txt` | Optional: Ultralytics and TensorFlow for YOLOv8 training and LSTM extensions | Section 4.7 |

## Running the notebooks

Run each notebook from its own module folder (paths are relative: `../../data/...`), in numeric order.
In Google Colab, upload the whole folder and keep its structure. PW01–PW14 run on an ordinary CPU in
about 30 seconds in total. PW15 checks the 40/10/10 image split; its YOLOv8 training cell is commented
out because it needs a GPU and internet access for pre-trained weights.

## Notes on numerical reproducibility

* **ARIMA (PW11).** The maximum-likelihood fit of ARIMA(2,1,2) does not fully converge on several
  windows (statsmodels prints `ConvergenceWarning`; this is used as a teaching point). Its MAE therefore
  depends slightly on NumPy/SciPy/statsmodels versions: 0.244–0.263 mg/L in our runs. The conclusion of
  Section 4.2 holds in every run: the 24-h seasonal naive baseline (MAE 0.222 mg/L, version-independent)
  is better than ARIMA.
* **Random forest (PW9).** Reported values (macro-F1 0.811, AP 0.810) are reproduced exactly with the
  pinned scikit-learn 1.5.2; newer releases give 0.818 / 0.797. Logistic regression is identical in all
  versions, so the conclusion (the more complex model is not better) is unchanged.
* All other values are reproduced exactly in both environments.

## Scope and evidence statement

These materials document the intervention. They do not by themselves demonstrate educational
effectiveness, transferability or scalability. All datasets in `data/` are synthetic and must not be
described as farm data. The CV images are programmatically drawn ellipses. Student outcome data are
not included.

## Licences

Code and original teaching text: MIT. Synthetic datasets: CC0. Third-party material in S1: see
`supplementary/S1_real_image_validation/ATTRIBUTION.md`.
