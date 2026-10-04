import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelduel import corrected_ttest, duel


def test_corrected_is_more_conservative_than_naive():
    d = np.random.default_rng(0).normal(0.01, 0.02, 50)
    p, lo, hi, naive_p = corrected_ttest(d, n_train=400, n_test=100)
    assert p > naive_p and lo < d.mean() < hi


def test_correction_formula():
    d = np.array([0.02, 0.01, 0.03, 0.00, 0.04])
    p, lo, hi, _ = corrected_ttest(d, n_train=80, n_test=20, alpha=0.05)
    se = np.sqrt((1 / 5 + 20 / 80) * d.var(ddof=1))
    assert np.isclose(hi - d.mean(), 2.776445 * se, rtol=1e-5)  # t(0.975, df=4)


def test_identical_models():
    X, y = load_breast_cancer(return_X_y=True)
    r = duel(DummyClassifier(), DummyClassifier(), X, y, n_repeats=2)
    assert r.diff == 0 and r.p_value == 1.0 and r.winner is None


def test_constant_nonzero_difference_is_not_called_significant():
    # same gap on every fold, plus float rounding noise: there's no spread to test against
    d = np.full(10, -1 / 15) + np.random.default_rng(0).normal(0, 1e-17, 10)
    p, lo, hi, _ = corrected_ttest(d, n_train=120, n_test=30)
    assert np.isnan(p)
    X, y = load_iris(return_X_y=True)  # reproduces the case found when installing from GitHub
    r = duel(DummyClassifier(), DummyClassifier(strategy="uniform", random_state=0), X, y, n_repeats=2)
    assert r.winner is None and "Can't assess" in str(r)


def test_detects_a_real_difference():
    X, y = load_breast_cancer(return_X_y=True)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    r = duel(lr, DummyClassifier(), X, y, n_repeats=3)
    assert r.winner == "A" and r.diff > 0.3
    assert "A is better" in str(r)


def test_regression_with_integer_targets_defaults_to_r2():
    X, y = load_diabetes(return_X_y=True)  # integer targets: must still be treated as regression
    r = duel(LinearRegression(), DummyRegressor(), X, y, n_repeats=2)
    assert r.scoring == "r2" and r.winner == "A"


def test_rejects_mixed_model_types():
    X, y = load_breast_cancer(return_X_y=True)
    try:
        duel(DummyClassifier(), DummyRegressor(), X, y)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
