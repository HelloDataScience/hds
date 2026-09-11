# 회귀 진단 함수 테스트
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sma
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oi
from scipy import stats

from hds import stat


# hat_matrix(), leverage()
def test_hat_matrix_does_not_modify_X(reg_data):
    _, X = reg_data
    before = X.copy()
    stat.hat_matrix(X=X)
    pd.testing.assert_frame_equal(X, before)


def test_leverage_does_not_modify_X(reg_data):
    _, X = reg_data
    before = X.copy()
    stat.leverage(X=X)
    pd.testing.assert_frame_equal(X, before)


def test_leverage_matches_statsmodels(reg_data):
    y, X = reg_data
    expected = stat.ols(y=y, X=X).get_influence().hat_matrix_diag
    result = stat.leverage(X=X)
    assert result.name == 'Leverage'
    assert result.is_monotonic_decreasing
    np.testing.assert_allclose(result.loc[X.index], expected)


def test_hat_matrix_diagonal_matches_leverage(reg_data):
    _, X = reg_data
    hat = stat.hat_matrix(X=X.to_numpy())
    np.testing.assert_allclose(np.diag(hat), stat.leverage(X=X).loc[X.index])


# std_resid()
def test_std_resid_keeps_original_index(reg_data):
    y, X = reg_data
    model = stat.ols(y=y, X=X)
    result = stat.std_resid(model=model)

    assert set(result.index) == set(model.resid.index)
    assert result.abs().is_monotonic_decreasing

    expected = stats.zscore(model.resid.to_numpy())
    np.testing.assert_allclose(result.loc[model.resid.index], expected)


# vif()
def test_vif_matches_statsmodels(reg_data):
    y, X = reg_data
    model = stat.ols(y=y, X=X)
    exog = model.model.exog
    expected = [
        oi.variance_inflation_factor(exog=exog, exog_idx=i)
        for i in range(1, 4)
    ]
    result = stat.vif(model=model)
    assert list(result.columns) == ['x1', 'x2', 'x3']
    np.testing.assert_allclose(result.iloc[0], expected)


def test_vif_excludes_formula_intercept(reg_data):
    y, X = reg_data
    data = pd.concat(objs=[X, y], axis=1)
    model = smf.ols(formula='y ~ x1 + x2 + x3', data=data).fit()
    assert list(stat.vif(model=model).columns) == ['x1', 'x2', 'x3']


def test_vif_without_const(reg_data):
    y, X = reg_data
    model = sma.OLS(endog=y, exog=X).fit()
    expected = [
        oi.variance_inflation_factor(exog=X.to_numpy(), exog_idx=i)
        for i in range(3)
    ]
    result = stat.vif(model=model)
    assert list(result.columns) == ['x1', 'x2', 'x3']
    np.testing.assert_allclose(result.iloc[0], expected)


# regression_diagnosis()
def test_regression_diagnosis_returns_figure_and_axes(reg_data):
    y, X = reg_data
    fig, axes = stat.regression_diagnosis(model=stat.ols(y=y, X=X))

    assert isinstance(fig, plt.Figure)
    assert axes.shape == (2, 2)

    titles = [ax.get_title() for ax in axes.flatten()[:3]]
    assert titles == ['Residuals vs Fitted', 'Normal Q-Q', 'Scale-Location']


def test_regression_diagnosis_does_not_draw_on_current_figure(reg_data):
    y, X = reg_data
    current = plt.figure()
    fig, _ = stat.regression_diagnosis(model=stat.ols(y=y, X=X))
    assert fig is not current
    assert current.axes == []


def test_regression_diagnosis_qq_line_covers_data(reg_data):
    y, X = reg_data
    y = y.copy()
    y.iloc[0] += 30  # 표준화 잔차가 4를 넘는 이상치
    _, axes = stat.regression_diagnosis(model=stat.ols(y=y, X=X))

    ax = axes[0, 1]
    points = ax.collections[0].get_offsets()
    line = ax.get_lines()[-1]

    assert points.max() > 4
    assert min(line.get_xdata()) == pytest.approx(points.min())
    assert max(line.get_xdata()) == pytest.approx(points.max())
