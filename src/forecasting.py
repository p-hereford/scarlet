import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


TARGET_COL = "fraud_complaints"
DATE_COL = "week"

"""
SPIKE_THRESHOLD was set at 100 because this is where the week stops looking like ordinary variation and
starts looking like a stress event worth treating separately.
"""
SPIKE_THRESHOLD = 100.0

NONZERO_THRESHOLD = 0.0

"""
These regularisation settings were kept tight because the series is short and I'd rather shrink a bit too much 
than let a small sample talk me into nonsense.
"""
CLF_C = 0.05
REG_ALPHA = 15.0

"""
The walk-forward setup was kept small and expanding because this is not a big enough series to waste history,
but it still needs to be tested in something resembling live use.
"""
CV_INITIAL_TRAIN = 38
CV_STEP = 5
CV_TEST_HORIZON = 5

"""
The feature set was kept deliberately narrow because on a series this short, adding more correlated lags usually
gives the model more ways to hallucinate rather than more signal.
"""
FEATURE_COLS = [
    f"{TARGET_COL}_lag1",
    f"{TARGET_COL}_roll_mean_4w_lag1",
    "industry_fraud_complaints_lag1",
    "industry_fraud_growth_lag1",
    "weeks_since_nonzero",
]

ROLL_MEAN_FEATURE = f"{TARGET_COL}_roll_mean_4w_lag1"


def _weeks_since_condition(values: pd.Series, condition_fn) -> pd.Series:
    out = []
    last_idx = None
    for i, v in enumerate(values):
        out.append(np.nan if last_idx is None else float(i - last_idx))
        if condition_fn(v):
            last_idx = i
    return pd.Series(out, index=values.index, dtype=float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Everything here is built from lagged information because I did not want even accidental leakage
    dressed up as clever feature engineering.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    shifted = df[TARGET_COL].shift(1)

    df[f"{TARGET_COL}_lag1"] = shifted
    df[ROLL_MEAN_FEATURE] = shifted.rolling(4, min_periods=4).mean()
    df["weeks_since_nonzero"] = (
        _weeks_since_condition(df[TARGET_COL], lambda v: v > NONZERO_THRESHOLD)
        .shift(1)
    )
    df["month"] = df[DATE_COL].dt.month

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


class HurdleForecaster:
    """
    I split this into classification and magnitude because the real question was not just how many complaints there might be,
    but whether the week had crossed into a different kind of event.
    """

    def __init__(self, clf_c: float = CLF_C, reg_alpha: float = REG_ALPHA) -> None:
        self.clf = LogisticRegression(C=clf_c, max_iter=1000, random_state=42)
        self.reg = Ridge(alpha=reg_alpha)
        self.scaler = StandardScaler()
        self._fitted = False
        self._roll_mean_idx = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "HurdleForecaster":
        x_scaled = self.scaler.fit_transform(features)

        """
        The classifier targets spike weeks rather than nonzero weeks because most weeks are already nonzero,
         so that target would be too broad to be useful.
        """
        y_spike = (targets >= SPIKE_THRESHOLD).astype(int)
        self.clf.fit(x_scaled, y_spike)

        spike_mask = targets >= SPIKE_THRESHOLD
        if spike_mask.sum() < 2:
            """
            If there are barely any spike weeks in the training window, it is safer to fall back to the full sample 
            than pretend the conditional fit means more than it does.
            """
            self.reg.fit(x_scaled, np.log1p(targets))
        else:
            """
            The regressor is trained on spike weeks only because once the week is clearly not a spike, the rolling baseline is 
            a better anchor than teaching the magnitude model to care about quiet weeks.
            """
            self.reg.fit(x_scaled[spike_mask], np.log1p(targets[spike_mask]))

        self._fitted = True
        return self

    def predict(
        self,
        features: np.ndarray,
        roll_mean_values: "np.ndarray | None" = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        x_scaled = self.scaler.transform(features)
        p_spike = self.clf.predict_proba(x_scaled)[:, 1]

        count_given_spike = np.expm1(self.reg.predict(x_scaled))
        count_given_spike = np.clip(count_given_spike, 0.0, None)

        """
        Non-spike weeks fall back to the rolling mean because forcing the conditional spike model to speak on calm weeks 
        would drag the forecast away from a steadier baseline.
        """
        fallback = (
            np.clip(roll_mean_values, 0.0, None)
            if roll_mean_values is not None
            else np.clip(features[:, 1], 0.0, None)
        )

        return p_spike * count_given_spike + (1.0 - p_spike) * fallback

    def predict_spike_proba(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(features))[:, 1]

    def predict_spike_binary(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        return (self.predict_spike_proba(features) >= threshold).astype(int)


def baseline_lag1(df: pd.DataFrame) -> np.ndarray:
    return df[f"{TARGET_COL}_lag1"].to_numpy(dtype=float)


def baseline_rolling_mean(df: pd.DataFrame) -> np.ndarray:
    return df[ROLL_MEAN_FEATURE].to_numpy(dtype=float)


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def subset_mean_absolute_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def spike_classification_metrics(
    actual: np.ndarray,
    spike_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_true = (actual >= SPIKE_THRESHOLD).astype(int)
    y_predicted = (spike_proba >= threshold).astype(int)

    if y_true.sum() == 0:
        return {"precision_spike": np.nan, "recall_spike": np.nan, "f1_spike": np.nan}

    return {
        "precision_spike": precision_score(y_true, y_predicted, zero_division=0),
        "recall_spike": recall_score(y_true, y_predicted, zero_division=0),
        "f1_spike": f1_score(y_true, y_predicted, zero_division=0),
    }


def summarize_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    name: str,
    spike_proba: "np.ndarray | None" = None,
) -> dict:
    spike_mask = actual >= SPIKE_THRESHOLD
    nonzero_mask = actual > NONZERO_THRESHOLD

    result = {
        "model": name,
        "mae_all": mean_absolute_error(actual, predicted),
        "rmse_all": root_mean_squared_error(actual, predicted),
        "mae_nonzero_weeks": subset_mean_absolute_error(actual, predicted, nonzero_mask),
        "mae_spike_weeks": subset_mean_absolute_error(actual, predicted, spike_mask),
    }

    if spike_proba is not None:
        result.update(spike_classification_metrics(actual, spike_proba))

    return result


def walk_forward_cv(
    df: pd.DataFrame,
    initial_train: int = CV_INITIAL_TRAIN,
    step: int = CV_STEP,
    test_horizon: int = CV_TEST_HORIZON,
) -> pd.DataFrame:
    """
    Walk-forward CV was used because a random split would flatter the model on a time series and a single hold-out
    would be too flimsy on something this short.
    """
    n = len(df)
    x_all = df[FEATURE_COLS].to_numpy(dtype=float)
    y_all = df[TARGET_COL].to_numpy(dtype=float)

    records = []
    train_end = initial_train
    fold = 0

    while train_end + test_horizon <= n:
        fold += 1
        test_end = train_end + test_horizon

        x_train = x_all[:train_end]
        y_train = y_all[:train_end]
        x_test = x_all[train_end:test_end]
        y_test = y_all[train_end:test_end]

        roll_test = df[ROLL_MEAN_FEATURE].iloc[train_end:test_end].to_numpy(float)
        lag1_test = df[f"{TARGET_COL}_lag1"].iloc[train_end:test_end].to_numpy(float)

        model = HurdleForecaster()
        model.fit(x_train, y_train)

        hurdle_predicted = model.predict(x_test, roll_mean_values=roll_test)
        hurdle_proba = model.predict_spike_proba(x_test)

        row = {"fold": fold, "train_weeks": train_end, "test_weeks": test_horizon}

        for key, value in summarize_metrics(y_test, hurdle_predicted, "hurdle", hurdle_proba).items():
            if key != "model":
                row[f"hurdle_{key}"] = value
        for key, value in summarize_metrics(y_test, lag1_test, "lag1").items():
            if key != "model":
                row[f"lag1_{key}"] = value
        for key, value in summarize_metrics(y_test, roll_test, "roll_mean").items():
            if key != "model":
                row[f"roll_{key}"] = value

        records.append(row)
        train_end += step

    return pd.DataFrame(records)


def final_evaluation(
    df: pd.DataFrame,
    test_frac: float = 0.20,
) -> "tuple[HurdleForecaster, pd.DataFrame, pd.DataFrame]":
    """
    The final hold-out is the most recent slice because that is the closest thing
    this little series has to a live deployment test.
    """
    n = len(df)
    split = int(n * (1 - test_frac))

    train_df = df.iloc[:split]
    test_df = df.iloc[split:].copy()

    x_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    x_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    model = HurdleForecaster()
    model.fit(x_train, y_train)

    roll_test = baseline_rolling_mean(test_df)
    hurdle_predicted = model.predict(x_test, roll_mean_values=roll_test)
    hurdle_proba = model.predict_spike_proba(x_test)
    lag1_predicted = baseline_lag1(test_df)

    results_df = test_df[[DATE_COL, TARGET_COL]].copy()
    results_df["p_spike"] = hurdle_proba
    results_df["pred_hurdle"] = hurdle_predicted
    results_df["pred_lag1"] = lag1_predicted
    results_df["pred_roll_mean"] = roll_test
    results_df["err_hurdle"] = np.abs(y_test - hurdle_predicted)
    results_df["err_lag1"] = np.abs(y_test - lag1_predicted)
    results_df["is_spike"] = y_test >= SPIKE_THRESHOLD

    metrics_df = pd.DataFrame([
        summarize_metrics(y_test, hurdle_predicted, "hurdle_model", hurdle_proba),
        summarize_metrics(y_test, lag1_predicted, "naive_lag1"),
        summarize_metrics(y_test, roll_test, "naive_roll_mean"),
    ])

    return model, results_df, metrics_df


def _dashed_row(col_widths: list) -> str:
    segments = []
    for w in col_widths:
        n = w - 2
        segments.append(" " + ("- " * (n // 2 + 1))[:n] + " ")
    return "|".join(segments)


def _data_row(cells: list, col_widths: list) -> str:
    parts = []
    for cell, w in zip(cells, col_widths):
        parts.append(f"{str(cell):^{w}}")
    return "|".join(parts)


def _render_grid(header_line: str, col_headers: list, row_blocks: list) -> None:
    n_cols = len(col_headers)
    col_content = [[] for _ in range(n_cols)]
    for i, h in enumerate(col_headers):
        col_content[i].append(h)
    for block in row_blocks:
        for row in block:
            for i, cell in enumerate(row):
                if i < n_cols:
                    col_content[i].append(str(cell))

    col_widths = [max(len(s) for s in cells) + 4 for cells in col_content]
    total_w    = sum(col_widths) + (n_cols - 1)

    print()
    print(f"{header_line:^{total_w}}")
    print()
    print(_data_row(col_headers, col_widths))
    print(_dashed_row(col_widths))

    for b_i, block in enumerate(row_blocks):
        for row in block:
            padded = list(row) + [""] * (n_cols - len(row))
            print(_data_row(padded, col_widths))
        if b_i < len(row_blocks) - 1:
            print(_dashed_row(col_widths))

    print()


def _get(metrics_df: pd.DataFrame, model: str, col: str) -> float:
    row = metrics_df[metrics_df["model"] == model]
    if row.empty:
        return float("nan")
    v = row.iloc[0].get(col, float("nan"))
    return float(v) if v is not None else float("nan")


def _f(v) -> str:
    return "--" if (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def _d(v, base) -> str:
    if any(isinstance(x, float) and np.isnan(x) for x in [v, base]):
        return ""
    return f"{v - base:+.2f}"


def print_performance_grid(
    metrics_df: pd.DataFrame,
    results_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    df: pd.DataFrame,
    now: str,
) -> None:
    dates = pd.to_datetime(results_df[DATE_COL])
    start = dates.min().strftime("%Y-%m-%d")
    end   = dates.max().strftime("%Y-%m-%d")

    header = (
        f"SCARLET  //  FRAUD EARLY WARNING ENGINE  //  PH RISK ANALYTICS"
        f"   {now}   {start} - {end}"
    )

    h_mae  = _get(metrics_df, "hurdle_model",    "mae_all")
    l_mae  = _get(metrics_df, "naive_lag1",      "mae_all")
    r_mae  = _get(metrics_df, "naive_roll_mean", "mae_all")

    h_mspk = _get(metrics_df, "hurdle_model",    "mae_spike_weeks")
    l_mspk = _get(metrics_df, "naive_lag1",      "mae_spike_weeks")
    r_mspk = _get(metrics_df, "naive_roll_mean", "mae_spike_weeks")

    h_mnz  = _get(metrics_df, "hurdle_model",    "mae_nonzero_weeks")
    l_mnz  = _get(metrics_df, "naive_lag1",      "mae_nonzero_weeks")
    r_mnz  = _get(metrics_df, "naive_roll_mean", "mae_nonzero_weeks")

    h_rmse = _get(metrics_df, "hurdle_model",    "rmse_all")
    l_rmse = _get(metrics_df, "naive_lag1",      "rmse_all")
    r_rmse = _get(metrics_df, "naive_roll_mean", "rmse_all")

    h_prec = _get(metrics_df, "hurdle_model",    "precision_spike")
    h_rec  = _get(metrics_df, "hurdle_model",    "recall_spike")
    h_f1   = _get(metrics_df, "hurdle_model",    "f1_spike")

    spike_df = results_df[results_df["is_spike"]]
    hi = int(df[TARGET_COL].max())
    av = int(df[TARGET_COL].mean())
    lo = int(df[TARGET_COL].min())
    ns = int((df[TARGET_COL] >= SPIKE_THRESHOLD).sum())
    nw = len(df)

    def cv(col):
        return cv_df[col].mean() if col in cv_df.columns else float("nan")

    cv_mae  = cv("hurdle_mae_all")
    cv_mspk = cv("hurdle_mae_spike_weeks")
    cv_rec  = cv("hurdle_recall_spike")
    cv_prec = cv("hurdle_precision_spike")
    nf      = len(cv_df)

    col_headers = ["", "HURDLE MODEL", "NAIVE  LAG-1", "ROLL  MEAN", "STATS"]

    row_blocks = [
        [
            ["MAE  (ALL WEEKS)",     _f(h_mae),  _f(l_mae),  _f(r_mae),  f"HI:  {hi}"],
            ["",                     _d(h_mae, l_mae), _d(l_mae, h_mae), _d(r_mae, l_mae), f"AV:  {av}"],
            ["",                     f"n={nw}",   f"n={nw}",   f"n={nw}",   f"LO:  {lo}"],
        ],
        [
            ["MAE  (STRESS WKS)",    _f(h_mspk), _f(l_mspk), _f(r_mspk), f"N STRESS: {ns}"],
            ["",                     _d(h_mspk, l_mspk), _d(l_mspk, h_mspk), _d(r_mspk, l_mspk), ""],
            ["",                     f"n={ns}",   f"n={ns}",   f"n={ns}",   ""],
        ],
        [
            ["MAE  (NONZERO WKS)",   _f(h_mnz),  _f(l_mnz),  _f(r_mnz),  ""],
            ["",                     _d(h_mnz, l_mnz), _d(l_mnz, h_mnz), _d(r_mnz, l_mnz), ""],
        ],
        [
            ["RMSE  (ALL WEEKS)",    _f(h_rmse), _f(l_rmse), _f(r_rmse), ""],
            ["",                     _d(h_rmse, l_rmse), _d(l_rmse, h_rmse), _d(r_rmse, l_rmse), ""],
        ],
        [
            ["PRECISION  (SPIKE)",   _f(h_prec), "--",        "--",        f"CV FOLDS: {nf}"],
            ["RECALL     (SPIKE)",   _f(h_rec),  "--",        "--",        f"CV MAE:   {_f(cv_mae)}"],
            ["F1         (SPIKE)",   _f(h_f1),   "--",        "--",        f"CV MSPK:  {_f(cv_mspk)}"],
            ["",                     "",          "",          "",          f"CV REC:   {_f(cv_rec)}"],
            ["",                     "",          "",          "",          f"CV PREC:  {_f(cv_prec)}"],
        ],
    ]

    _render_grid(header, col_headers, row_blocks)


def print_weekly_grid(results_df: pd.DataFrame) -> None:
    header = "WEEKLY FORECAST DETAIL  //  HOLD-OUT PERIOD"

    col_headers = ["WEEK", "ACTUAL", "P(SPIKE)", "SCARLET", "LAG-1",
                   "ERR SCARLET", "ERR LAG-1", "FLAG"]

    rows = []
    for _, r in results_df.iterrows():
        rows.append([
            str(r[DATE_COL])[:10],
            int(r[TARGET_COL]),
            f"{r['p_spike']:.3f}",
            f"{r['pred_hurdle']:.1f}",
            f"{r['pred_lag1']:.1f}",
            f"{r['err_hurdle']:.1f}",
            f"{r['err_lag1']:.1f}",
            "** STRESS **" if r["is_spike"] else "",
        ])

    _render_grid(header, col_headers, [rows])


def print_stress_grid(results_df: pd.DataFrame) -> None:
    spike_df = results_df[results_df["is_spike"]]
    if spike_df.empty:
        return

    header = f"STRESS EVENTS  //  WEEKS AT OR ABOVE {SPIKE_THRESHOLD:.0f}"
    col_headers = ["WEEK", "ACTUAL", "P(SPIKE)", "SCARLET", "LAG-1",
                   "ERR SCARLET", "ERR LAG-1"]

    rows = []
    for _, r in spike_df.iterrows():
        rows.append([
            str(r[DATE_COL])[:10],
            int(r[TARGET_COL]),
            f"{r['p_spike']:.3f}",
            f"{r['pred_hurdle']:.1f}",
            f"{r['pred_lag1']:.1f}",
            f"{r['err_hurdle']:.1f}",
            f"{r['err_lag1']:.1f}",
        ])

    summary_block = [
        ["SCARLET  MAE  (STRESS EVENTS)", f"{spike_df['err_hurdle'].mean():.2f}", "", "", "", "", ""],
        ["LAG-1    MAE  (STRESS EVENTS)", f"{spike_df['err_lag1'].mean():.2f}",   "", "", "", "", ""],
    ]

    _render_grid(header, col_headers, [rows, summary_block])


def print_signal_grid(model: HurdleForecaster) -> None:
    header = "SIGNAL WEIGHTS  //  LOGISTIC CLASSIFIER  +  RIDGE REGRESSOR"
    col_headers = ["FEATURE", "CLF COEFF", "CLF DIRECTION", "REG COEFF"]

    coef_df = pd.DataFrame({
        "feature":  FEATURE_COLS,
        "coef_clf": model.clf.coef_[0],
        "coef_reg": model.reg.coef_,
    }).sort_values("coef_clf", key=abs, ascending=False)

    rows = []
    for _, r in coef_df.iterrows():
        direction = "RAISES RISK" if r["coef_clf"] > 0 else "LOWERS RISK"
        rows.append([
            r["feature"].upper(),
            f"{r['coef_clf']:+.4f}",
            direction,
            f"{r['coef_reg']:+.4f}",
        ])

    _render_grid(header, col_headers, [rows])


def main() -> None:
    now = datetime.now().strftime("%d %b %Y  %H:%M")

    df = pd.read_csv("data/jpm_weekly_features.csv")
    df = build_features(df)

    cv_df                         = walk_forward_cv(df)
    model, results_df, metrics_df = final_evaluation(df)

    print_performance_grid(metrics_df, results_df, cv_df, df, now)
    print_weekly_grid(results_df)
    print_stress_grid(results_df)
    print_signal_grid(model)


if __name__ == "__main__":
    main()