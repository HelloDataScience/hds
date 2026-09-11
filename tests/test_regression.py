# 회귀 모델 적합 함수 테스트
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sma
from sklearn.linear_model import LinearRegression, LogisticRegression

from hds import stat

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ['forward', 'backward', 'both']


# ols(), glm()
def test_ols_does_not_modify_X(reg_data):
    y, X = reg_data
    before = X.copy()
    stat.ols(y=y, X=X)
    pd.testing.assert_frame_equal(X, before)


def test_glm_does_not_modify_X(logit_data):
    y, X = logit_data
    before = X.copy()
    stat.glm(y=y, X=X)
    pd.testing.assert_frame_equal(X, before)


def test_ols_matches_statsmodels(reg_data):
    y, X = reg_data
    expected = sma.OLS(endog=y, exog=sma.add_constant(X)).fit()
    model = stat.ols(y=y, X=X)
    pd.testing.assert_series_equal(model.params, expected.params)


def test_glm_matches_statsmodels(logit_data):
    y, X = logit_data
    expected = sma.GLM(
        endog=y,
        exog=sma.add_constant(X),
        family=sma.families.Binomial(),
    ).fit()
    model = stat.glm(y=y, X=X)
    pd.testing.assert_series_equal(model.params, expected.params)


def test_ols_accepts_ndarray(reg_data):
    y, X = reg_data
    expected = stat.ols(y=y, X=X)
    model = stat.ols(y=y.to_numpy(), X=X.to_numpy())
    assert model.model.exog_names == ['const', 'x1', 'x2', 'x3']
    np.testing.assert_allclose(model.params, expected.params)


def test_ols_ndarray_X_uses_index_of_y(reg_data):
    y, X = reg_data
    model = stat.ols(y=y, X=X.to_numpy())
    assert model.resid.index.equals(y.index)


def test_ols_accepts_list_y(reg_data):
    y, X = reg_data
    expected = stat.ols(y=y, X=X)
    model = stat.ols(y=y.tolist(), X=X.to_numpy())
    np.testing.assert_allclose(model.params, expected.params)


def test_ols_does_not_duplicate_const(reg_data):
    y, X = reg_data
    model = stat.ols(y=y, X=sma.add_constant(X))
    assert model.model.exog_names.count('const') == 1


# coefs(), std_coefs()
def test_coefs_statsmodels(reg_data):
    y, X = reg_data
    model = stat.ols(y=y, X=X)
    pd.testing.assert_series_equal(stat.coefs(model=model), model.params)


def test_coefs_sklearn_regression(reg_data):
    y, X = reg_data
    model = LinearRegression().fit(X=X, y=y)
    result = stat.coefs(model=model)
    assert list(result.index) == ['x1', 'x2', 'x3']
    np.testing.assert_allclose(result, model.coef_)


def test_coefs_sklearn_without_feature_names(reg_data):
    y, X = reg_data
    model = LinearRegression().fit(X=X.to_numpy(), y=y)
    assert list(stat.coefs(model=model).index) == ['x0', 'x1', 'x2']


def test_coefs_sklearn_binary_classifier(logit_data):
    y, X = logit_data
    model = LogisticRegression().fit(X=X, y=y)
    result = stat.coefs(model=model)
    assert isinstance(result, pd.Series)
    np.testing.assert_allclose(result, model.coef_[0])


def test_coefs_sklearn_multiclass(reg_data):
    y, X = reg_data
    labels = np.digitize(y, bins=np.quantile(y, [1 / 3, 2 / 3]))
    model = LogisticRegression().fit(X=X, y=labels)
    result = stat.coefs(model=model)
    assert result.shape == (3, 3)
    assert list(result.index) == [0, 1, 2]


def test_coefs_unsupported_model():
    with pytest.raises(TypeError):
        stat.coefs(model=object())


def test_std_coefs_supported_models(reg_data, logit_data):
    y, X = reg_data
    result = stat.std_coefs(model=stat.ols(y=y, X=X))
    assert list(result.index) == ['const', 'x1', 'x2', 'x3']

    y, X = logit_data
    result = stat.std_coefs(model=stat.glm(y=y, X=X))
    assert list(result.index) == ['const', 'x1', 'x2']


def test_std_coefs_matches_standardized_regression(reg_data):
    y, X = reg_data
    result = stat.std_coefs(model=stat.ols(y=y, X=X))

    # 입력변수와 목표변수를 표준화하여 적합한 회귀계수와 같아야 함
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    expected = sma.OLS(endog=y_std, exog=sma.add_constant(X_std)).fit()

    names = ['x1', 'x2', 'x3']
    np.testing.assert_allclose(result[names], expected.params[names])


def test_std_coefs_glm_scales_only_inputs(logit_data):
    y, X = logit_data
    model = stat.glm(y=y, X=X)
    result = stat.std_coefs(model=model)

    names = ['x1', 'x2']
    expected = model.params[names] * X[names].std()
    np.testing.assert_allclose(result[names], expected)


def test_std_coefs_unsupported_model(logit_data):
    y, X = logit_data
    model = sma.Logit(endog=y, exog=sma.add_constant(X)).fit(disp=0)
    with pytest.raises(TypeError):
        stat.std_coefs(model=model)


# stepwise()
@pytest.mark.parametrize('direction', DIRECTIONS)
def test_stepwise_does_not_modify_X(reg_data, direction):
    y, X = reg_data
    before = X.copy()
    stat.stepwise(y=y, X=X, direction=direction)
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_stepwise_selects_true_variables(reg_data, direction):
    y, X = reg_data
    model = stat.stepwise(y=y, X=X, direction=direction)
    assert {'x1', 'x2'} <= set(model.params.index)


def test_stepwise_invalid_direction(reg_data):
    y, X = reg_data
    with pytest.raises(ValueError, match='direction'):
        stat.stepwise(y=y, X=X, direction='foward')


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_stepwise_handles_names_with_spaces(reg_data, direction):
    y, X = reg_data
    X = X.rename(columns={'x1': 'x 1'})
    model = stat.stepwise(y=y.rename('price usd'), X=X, direction=direction)
    assert "Q('x 1')" in model.params.index


def test_stepwise_requires_dataframe(reg_data):
    y, X = reg_data
    with pytest.raises(TypeError):
        stat.stepwise(y=y, X=X.to_numpy())


def test_forward_selection_without_selected_variables():
    # y와 x1이 정확히 직교하므로 x1을 추가해도 AIC가 낮아지지 않음
    y = pd.Series([1.0, -1.0, 1.0, -1.0] * 5, name='y')
    X = pd.DataFrame(data={'x1': [1.0, 1.0, -1.0, -1.0] * 5})
    model = stat.stepwise(y=y, X=X, direction='forward')
    assert list(model.params.index) == ['Intercept']


# 모든 입력변수가 목표변수와 관련 있어 후진소거법이 변수를 모두 남기는 데이터
ORDER_DATA = '''
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=0)
names = ['x6', 'x5', 'x4', 'x3', 'x2', 'x1']
X = pd.DataFrame(data=rng.normal(size=(300, 6)), columns=names)
y = pd.Series(X.to_numpy() @ np.arange(1, 7) + rng.normal(size=300), name='y')
'''

ORDER_SCRIPT = ORDER_DATA + '''
from hds import stat

for direction in ['forward', 'backward', 'both']:
    model = stat.stepwise(y=y, X=X, direction=direction)
    print(direction, list(model.params.index))
'''


def test_backward_selection_keeps_column_order():
    namespace = {}
    exec(ORDER_DATA, namespace)
    X, y = namespace['X'], namespace['y']
    model = stat.stepwise(y=y, X=X, direction='backward')
    assert list(model.params.index) == ['Intercept'] + list(X.columns)


def test_stepwise_result_does_not_depend_on_hash_seed():
    outputs = []
    for seed in ['0', '1', '2', '3']:
        env = {
            **os.environ,
            'PYTHONHASHSEED': seed,
            'PYTHONPATH': str(ROOT),
            'MPLBACKEND': 'Agg',
        }
        completed = subprocess.run(
            [sys.executable, '-c', ORDER_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(completed.stdout)

    assert len(set(outputs)) == 1


# ols_table(), logit_table()
def test_ols_table_matches_model(reg_data):
    y, X = reg_data
    model = stat.ols(y=y, X=X)
    table = stat.ols_table(model=model)

    columns = ['coef', 'std_err', 't', 'p_value', 'ci_lower', 'ci_upper']
    assert list(table.columns) == columns
    assert list(table.index) == ['const', 'x1', 'x2', 'x3']

    np.testing.assert_allclose(table['coef'], model.params)
    np.testing.assert_allclose(table['std_err'], model.bse)
    np.testing.assert_allclose(table['t'], model.tvalues)
    np.testing.assert_allclose(table['p_value'], model.pvalues)
    np.testing.assert_allclose(
        table[['ci_lower', 'ci_upper']],
        model.conf_int(alpha=0.05),
    )


def test_ols_table_alpha(reg_data):
    y, X = reg_data
    model = stat.ols(y=y, X=X)
    table = stat.ols_table(model=model, alpha=0.1)
    np.testing.assert_allclose(
        table[['ci_lower', 'ci_upper']],
        model.conf_int(alpha=0.1),
    )


def test_ols_table_formula_and_robust_models(reg_data):
    y, X = reg_data
    model = stat.stepwise(y=y, X=X, direction='forward')
    assert stat.ols_table(model=model).index[0] == 'Intercept'

    robust = sma.OLS(endog=y, exog=sma.add_constant(X)).fit(cov_type='HC3')
    table = stat.ols_table(model=robust)
    assert 'z' in table.columns
    np.testing.assert_allclose(table['std_err'], robust.bse)


def test_ols_table_rejects_other_models(logit_data):
    y, X = logit_data
    with pytest.raises(TypeError):
        stat.ols_table(model=stat.glm(y=y, X=X))
    with pytest.raises(TypeError):
        stat.ols_table(model=LinearRegression().fit(X=X, y=y))


@pytest.mark.parametrize('alpha', [0, 1, 1.5])
def test_tables_reject_invalid_alpha(reg_data, logit_data, alpha):
    y, X = reg_data
    with pytest.raises(ValueError, match='alpha'):
        stat.ols_table(model=stat.ols(y=y, X=X), alpha=alpha)

    y, X = logit_data
    with pytest.raises(ValueError, match='alpha'):
        stat.logit_table(model=stat.glm(y=y, X=X), alpha=alpha)


def test_logit_table_odds_ratio(logit_data):
    y, X = logit_data
    model = stat.glm(y=y, X=X)
    table = stat.logit_table(model=model)
    ci = model.conf_int(alpha=0.05)

    columns = [
        'coef', 'std_err', 'z', 'p_value',
        'odds_ratio', 'or_ci_lower', 'or_ci_upper',
    ]
    assert list(table.columns) == columns
    assert list(table.index) == ['const', 'x1', 'x2']

    np.testing.assert_allclose(table['coef'], model.params)
    np.testing.assert_allclose(table['p_value'], model.pvalues)
    np.testing.assert_allclose(table['odds_ratio'], np.exp(model.params))
    np.testing.assert_allclose(table['or_ci_lower'], np.exp(ci[0]))
    np.testing.assert_allclose(table['or_ci_upper'], np.exp(ci[1]))


def test_logit_table_accepts_statsmodels_logit(logit_data):
    y, X = logit_data
    expected = stat.logit_table(model=stat.glm(y=y, X=X))
    model = sma.Logit(endog=y, exog=sma.add_constant(X)).fit(disp=0)
    table = stat.logit_table(model=model)
    np.testing.assert_allclose(
        table['odds_ratio'],
        expected['odds_ratio'],
        rtol=1e-4,
    )


def test_logit_table_rejects_other_models(reg_data, logit_data):
    y, X = reg_data
    with pytest.raises(TypeError):
        stat.logit_table(model=stat.ols(y=y, X=X))

    y, X = logit_data
    poisson = sma.GLM(
        endog=y,
        exog=sma.add_constant(X),
        family=sma.families.Poisson(),
    ).fit()
    with pytest.raises(TypeError):
        stat.logit_table(model=poisson)
