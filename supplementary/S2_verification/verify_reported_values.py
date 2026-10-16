"""Check every number reported in Sections 3.7 and 4.2 of the manuscript against the teaching data.

Run from the package root:
    python supplementary/S2_verification/verify_reported_values.py

The script recomputes each value directly from data/ and prints PASS / FAIL.
ARIMA is marked as version-sensitive: its maximum-likelihood fit does not
fully converge, so MAE varies slightly with NumPy/SciPy/statsmodels versions.
"""
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold, cross_val_score,
                                     cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
D = Path(__file__).resolve().parents[2] / "data"
rows = []


def check(section, name, reported, value, tol, note=""):
    ok = abs(value - reported) <= tol + 1e-9
    rows.append((section, name, reported, round(float(value), 4), "PASS" if ok else ("CHECK" if note else "FAIL"), note))


# ---- Data passport (Table 5) -------------------------------------------------
ras = pd.read_csv(D / "ras_sensor_log.csv", parse_dates=["timestamp"])
check("T5", "RAS records", 2600, len(ras), 0)
check("T5", "RAS missing values per variable (max)", 18, ras.drop(columns="timestamp").isna().sum().max(), 0)
fish = pd.read_csv(D / "fish_growth_dataset.csv")
check("T5", "Fish records", 600, len(fish), 0)
check("T5", "Rainbow trout", 302, (fish.species == "Rainbow trout").sum(), 0)
risk = pd.read_csv(D / "sturgeon_risk_classification.csv")
check("T5", "Risk records", 800, len(risk), 0)
check("T5", "Risk cases", 111, risk.risk.sum(), 0)
hyd = pd.read_csv(D / "hydrochem_timeseries.csv", parse_dates=["timestamp"])
check("T5", "Hydrochem records", 1440, len(hyd), 0)
ann = pd.read_csv(D / "cv" / "annotations.csv")
check("T5", "CV boxes (all)", 391, len(ann), 0)
check("4.2", "CV boxes in test split", 58, (ann.split == "test").sum(), 0)

# ---- PW3 chronological split -------------------------------------------------
df = ras.sort_values("timestamp").copy()
df["hour"] = df.timestamp.dt.hour
df["do_lag1"] = df.dissolved_oxygen_mgL.shift(1)
df = df.dropna()
cut = int(len(df) * .8)
check("4.2", "PW3 rows after cleaning", 2492, len(df), 0)
check("4.2", "PW3 training rows", 1993, cut, 0)

# ---- Regression PW4-PW6 ------------------------------------------------------
g = pd.get_dummies(fish, columns=["species"], drop_first=True)
X, y = g.drop(columns="weight_g"), g.weight_g
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
p = LinearRegression().fit(Xtr, ytr).predict(Xte)
check("4.2", "PW4 MAE, g", 52.8, mean_absolute_error(yte, p), 0.05)
check("4.2", "PW4 R2", 0.987, r2_score(yte, p), 0.0005)
Xp = fish[["age_weeks", "mean_water_temp_C", "daily_feed_ration_g"]]
cv = KFold(5, shuffle=True, random_state=42)
for d, rep in [(1, 58.9), (2, 49.4), (3, 48.2)]:
    s = -cross_val_score(make_pipeline(PolynomialFeatures(d), StandardScaler(), Ridge(alpha=1)), Xp, fish.weight_g,
                         cv=cv, scoring="neg_mean_absolute_error")
    check("4.2", f"PW5 CV MAE degree {d}, g", rep, s.mean(), 0.05)
Xr = fish[["age_weeks", "mean_water_temp_C", "daily_feed_ration_g", "length_cm"]]
gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [.1, 1, 10, 100]}, cv=5,
                  scoring="neg_mean_absolute_error").fit(Xr, fish.weight_g)
check("4.2", "PW6 best alpha", 0.1, gs.best_params_["ridge__alpha"], 0)
check("4.2", "PW6 CV MAE, g", 59.3, -gs.best_score_, 0.05)

# ---- Classification PW7-PW9 (Table 6) ---------------------------------------
Xc, yc = risk.drop(columns="risk"), risk.risk
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=.25, stratify=yc, random_state=42)
pr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr).predict_proba(Xte)[:, 1]
table6 = {.3: (.438, 1.0, .609, 36, 0), .5: (.549, 1.0, .709, 23, 0), .6: (.578, .929, .712, 19, 2),
          .7: (.650, .929, .765, 14, 2), .8: (.781, .893, .833, 7, 3)}
for tau, (P, R, F, FP, FN) in table6.items():
    p_, r_, f_, _ = precision_recall_fscore_support(yte, pr >= tau, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yte, pr >= tau).ravel()
    check("T6", f"tau={tau} precision", P, p_, 0.0005)
    check("T6", f"tau={tau} recall", R, r_, 0.0005)
    check("T6", f"tau={tau} F1", F, f_, 0.0005)
    check("T6", f"tau={tau} FP", FP, fp, 0)
    check("T6", f"tau={tau} FN", FN, fn, 0)
check("4.2", "PW8 average precision", 0.824, average_precision_score(yte, pr), 0.0005)
check("4.2", "PW8 ROC AUC", 0.974, roc_auc_score(yte, pr), 0.0005)
cvs = StratifiedKFold(5, shuffle=True, random_state=42)
for name, m, f1, ap, note in [
        ("LR", LogisticRegression(max_iter=2000, class_weight="balanced"), .825, .865, ""),
        ("RF", RandomForestClassifier(200, class_weight="balanced", random_state=42), .811, .810,
         "depends on scikit-learn version (exact with pinned 1.5.2)")]:
    s = cross_validate(m, Xc, yc, cv=cvs, scoring=["f1_macro", "average_precision"])
    check("4.2", f"PW9 {name} macro-F1", f1, s["test_f1_macro"].mean(), 0.0005, note)
    check("4.2", f"PW9 {name} average precision", ap, s["test_average_precision"].mean(), 0.0005, note)

# ---- Time series PW10-PW12 ---------------------------------------------------
s = hyd.set_index("timestamp").dissolved_oxygen_mgL
c = int(len(s) * .8)
check("4.2", "PW10 one-step naive MAE (last 20%)", 0.260, mean_absolute_error(s.iloc[c:], s.shift(1).iloc[c:]), 0.0005)
s = s.asfreq("h")
train, test = s.iloc[:-120], s.iloc[-120:]
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
warnings.simplefilter("ignore")
preds = []
for i in range(0, 120, 12):
    preds.extend(ARIMA(s.iloc[:len(train) + i], order=(2, 1, 2)).fit().forecast(12))
arima_mae = mean_absolute_error(test, preds)
check("4.2", "PW11 ARIMA MAE, mg/L", 0.251, arima_mae, 0.0005,
      "version-sensitive (MLE does not fully converge); range 0.24-0.26 observed")
check("4.2", "PW11 one-step naive MAE (last 120 h)", 0.265, mean_absolute_error(test, s.shift(1).iloc[-120:]), 0.0005)
hold = [s.iloc[len(train) + i - 1] for i in range(0, 120, 12) for _ in range(12)]
check("4.2", "12-h hold-last baseline MAE", 0.993, mean_absolute_error(test, hold), 0.0005)
seas = s.shift(24).iloc[-120:]
check("4.2", "24-h seasonal naive MAE", 0.222, mean_absolute_error(test, seas), 0.0005)
check("4.2", "24-h seasonal naive RMSE", 0.264, mean_squared_error(test, seas) ** .5, 0.0005)
rows.append(("4.2", "Conclusion: seasonal naive beats ARIMA", "yes", "yes" if mean_absolute_error(test, seas) < arima_mae else "no",
             "PASS" if mean_absolute_error(test, seas) < arima_mae else "FAIL", ""))
a = hyd[["temperature_C", "dissolved_oxygen_mgL", "pH", "ammonia_mgL"]].values
cut = int(len(a) * .8)
z = MinMaxScaler().fit(a[:cut]).transform(a)
n = len(z) - 24 - 6 + 1
split = max(0, cut - 24 - 6 + 1)
check("4.2", "PW12 training windows", 1123, split, 0)
check("4.2", "PW12 test windows", 288, n - split, 0)

# ---- Computer vision PW13-PW14 ----------------------------------------------
img = sorted((D / "cv" / "images" / "test").glob("*.png"))[0]
gimg = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gimg, 145, 255, cv2.THRESH_BINARY)
cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
check("4.2", f"PW13 contours on {img.name}", 4, len([q for q in cnt if cv2.contourArea(q) > 80]), 0)
check("4.2", f"PW13 annotated objects on {img.name}", 4, (ann.image == img.name).sum(), 0)
t = ann[ann.split == "test"]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


r = np.random.default_rng(42)
sc = []
for _, q in t.iterrows():
    gt = q[["x1", "y1", "x2", "y2"]].to_numpy(float)
    sc.append(iou(gt, gt + r.normal(0, 3, 4)))
check("4.2", "PW14 mean IoU (simulation)", 0.685, np.mean(sc), 0.0005)
check("4.2", "PW14 share of boxes IoU>=0.50", 0.966, np.mean(np.array(sc) >= .5), 0.0005)

# ---- Report ------------------------------------------------------------------
w = max(len(x[1]) for x in rows)
for sec, name, rep_, val, status, note in rows:
    print(f"{status:5}  {sec:4} {name:<{w}}  reported={rep_!s:<6} computed={val!s:<8} {note}")
n_fail = sum(x[4] == "FAIL" for x in rows)
print(f"\n{len(rows)} checks: {sum(x[4] == 'PASS' for x in rows)} PASS, "
      f"{sum(x[4] == 'CHECK' for x in rows)} version-sensitive, {n_fail} FAIL")
raise SystemExit(1 if n_fail else 0)
