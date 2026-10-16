# Supplementary File S2 — computational verification

| File | Content |
|---|---|
| `notebook_verification_log.json` | End-to-end execution of all 15 notebooks from a clean copy of the package, in two environments: (A) the versions pinned in `requirements.txt`, (B) current library releases. For each notebook: status, run time and printed output. Result: 15/15 executed in both. |
| `executed_notebooks/` | The 15 notebooks with their outputs (environment A), so the results can be inspected without running code. |
| `verify_reported_values.py` | Recomputes 62 values reported in the manuscript (Table 5, Section 4.2, Table 6) directly from `data/` and prints PASS / FAIL. Run from the package root. |
| `verify_reported_values_output_A_pinned.txt` | Output in environment A: 61 PASS, 1 version-sensitive (ARIMA), 0 FAIL. |
| `verify_reported_values_output_B_current.txt` | Output in environment B: 59 PASS, 3 version-sensitive (ARIMA; random forest in PW9), 0 FAIL. |

Version-sensitive values and why they vary are explained in the main `README.md`. Library versions of each
run are recorded in the `environment` field of the log.
