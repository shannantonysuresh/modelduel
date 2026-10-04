"""Is model A really better than model B? Repeated cross-validation + the corrected resampled t-test.

Plain t-tests on CV fold scores treat folds as independent, but folds share most of their training data,
so the variance is underestimated and noise gets called "significant". Nadeau & Bengio (2003) correct the
variance by (1/k + n_test/n_train); that is what this module uses.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.base import clone, is_classifier
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score


@dataclass(frozen=True)
class DuelResult:
    scoring: str
    mean_a: float
    mean_b: float
    diff: float          # mean of (score_a - score_b) over all folds; higher score = better
    ci_low: float
    ci_high: float
    p_value: float       # corrected resampled t-test (use this)
    naive_p_value: float  # plain paired t-test on folds (shown only to illustrate the problem)
    alpha: float
    n_folds: int

    @property
    def significant(self):
        return bool(self.p_value < self.alpha)  # NaN (can't assess) -> False

    @property
    def winner(self):
        return None if not self.significant else ("A" if self.diff > 0 else "B")

    def __str__(self):
        head = (f"A: {self.mean_a:.4f}   B: {self.mean_b:.4f}   ({self.scoring}, {self.n_folds} folds)\n"
                f"A - B = {self.diff:+.4f}  ({1 - self.alpha:.0%} CI {self.ci_low:+.4f} to {self.ci_high:+.4f}),"
                f"  corrected p = {self.p_value:.3g}\n")
        if np.isnan(self.p_value):
            return head + ("-> Can't assess: A - B was identical on every fold, so the noise can't be estimated. "
                           "Usually a fixed-seed or degenerate model; this is not evidence either way.")
        if self.winner:
            return head + f"-> {self.winner} is better (significant at {self.alpha})."
        verdict = "-> No significant difference: this gap is within cross-validation noise."
        if self.naive_p_value < self.alpha:
            verdict += (f"\n   (A naive t-test on the folds would say p = {self.naive_p_value:.3g}: "
                        f"it ignores that folds share training data and overstates the evidence.)")
        return head + verdict


def corrected_ttest(diffs, n_train, n_test, alpha=0.05):
    """Nadeau & Bengio corrected resampled t-test on per-fold score differences.
    Returns (p_value, ci_low, ci_high, naive_p_value)."""
    d = np.asarray(diffs, float)
    k = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    if var <= 1e-12 * max(1.0, mean * mean):  # same difference on every fold (allowing float rounding)
        p = 1.0 if np.isclose(mean, 0) else np.nan  # nonzero but no spread: noise can't be estimated
        return p, mean, mean, p
    se = np.sqrt((1 / k + n_test / n_train) * var)
    naive_se = np.sqrt(var / k)
    p = 2 * stats.t.sf(abs(mean / se), k - 1)
    naive_p = 2 * stats.t.sf(abs(mean / naive_se), k - 1)
    half = stats.t.ppf(1 - alpha / 2, k - 1) * se
    return p, mean - half, mean + half, naive_p


def duel(model_a, model_b, X, y, scoring=None, n_splits=5, n_repeats=10, alpha=0.05, random_state=0, n_jobs=None):
    """Compare two scikit-learn estimators on identical repeated-CV splits.

    scoring: any scikit-learn scorer name ("accuracy", "roc_auc", "neg_mean_absolute_error", ...);
             defaults to accuracy for classification and R^2 for regression. Higher is always better.
    """
    discrete = is_classifier(model_a)  # not type_of_target(y): integer regression targets look "multiclass"
    if discrete != is_classifier(model_b):
        raise ValueError("model_a and model_b must both be classifiers or both be regressors")
    scoring = scoring or ("accuracy" if discrete else "r2")
    splitter = (RepeatedStratifiedKFold if discrete else RepeatedKFold)(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    a = cross_val_score(clone(model_a), X, y, scoring=scoring, cv=splitter, n_jobs=n_jobs)
    b = cross_val_score(clone(model_b), X, y, scoring=scoring, cv=splitter, n_jobs=n_jobs)
    n = len(y)
    n_test = n / n_splits
    p, lo, hi, naive_p = corrected_ttest(a - b, n - n_test, n_test, alpha)
    return DuelResult(scoring, a.mean(), b.mean(), (a - b).mean(), lo, hi, p, naive_p, alpha, len(a))
