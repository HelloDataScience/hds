# A/B 테스트 표본 크기·검정력·최소 효과 함수 테스트
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from hds import stat


# 결과표에서 항목 값을 꺼내는 함수
def value(table, item):
    return table.set_index('item').loc[item, 'value']


# ab_sample_size()
def test_sample_size_proportion_matches_statsmodels():
    h = proportion_effectsize(prop1=0.12, prop2=0.10)
    n = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=1)
    table = stat.ab_sample_size(baseline=0.10, lift=0.02)
    assert value(table, 'n_control') == int(np.ceil(n))
    assert value(table, 'n_treatment') == int(np.ceil(n))
    assert value(table, 'n_total') == 2 * int(np.ceil(n))


def test_sample_size_returns_table_columns():
    table = stat.ab_sample_size(baseline=0.10, lift=0.02)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ['item', 'value', 'description']
    assert isinstance(value(table, 'n_control'), int)


def test_sample_size_relative_lift_equals_absolute():
    absolute = stat.ab_sample_size(baseline=0.10, lift=0.02)
    relative = stat.ab_sample_size(baseline=0.10, lift=0.2, relative=True)
    assert value(absolute, 'n_control') == value(relative, 'n_control')


def test_sample_size_mean_matches_statsmodels():
    n = TTestIndPower().solve_power(effect_size=1000 / 25000, alpha=0.05, power=0.8, ratio=1)
    table = stat.ab_sample_size(metric='mean', baseline=13000, lift=1000, sd=25000)
    assert value(table, 'n_control') == int(np.ceil(n))


def test_sample_size_ratio_sets_treatment_size():
    h = proportion_effectsize(prop1=0.12, prop2=0.10)
    n = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=4)
    table = stat.ab_sample_size(baseline=0.10, lift=0.02, ratio=4)
    assert value(table, 'n_control') == int(np.ceil(n))
    assert value(table, 'n_treatment') == int(np.ceil(n * 4))


def test_sample_size_multiple_groups_use_bonferroni():
    h = proportion_effectsize(prop1=0.12, prop2=0.10)
    n = NormalIndPower().solve_power(effect_size=h, alpha=0.05 / 2, power=0.8, ratio=1)
    table = stat.ab_sample_size(baseline=0.10, lift=0.02, n_groups=3)
    assert value(table, 'alpha') == 0.025
    assert value(table, 'n_control') == int(np.ceil(n))
    assert value(table, 'n_total') == 3 * int(np.ceil(n))


def test_sample_size_cluster_design_effect():
    base = stat.ab_sample_size(baseline=0.10, lift=0.02)
    h = proportion_effectsize(prop1=0.12, prop2=0.10)
    n = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=1)
    table = stat.ab_sample_size(baseline=0.10, lift=0.02, icc=0.01, cluster_size=500)
    deff = 1 + (500 - 1) * 0.01
    assert value(table, 'design_effect') == round(deff, 4)
    assert value(table, 'n_control') == int(np.ceil(n * deff))
    assert value(table, 'n_control') > value(base, 'n_control')
    assert value(table, 'clusters_control') == int(np.ceil(value(table, 'n_control') / 500))


def test_sample_size_days_from_daily_traffic():
    table = stat.ab_sample_size(baseline=0.10, lift=0.02, daily_traffic=1000)
    assert value(table, 'days') == int(np.ceil(value(table, 'n_total') / 1000))


def test_sample_size_one_sided_needs_fewer():
    two = stat.ab_sample_size(baseline=0.10, lift=0.02)
    one = stat.ab_sample_size(baseline=0.10, lift=0.02, alternative='larger')
    assert value(one, 'n_control') < value(two, 'n_control')


# ab_power()
def test_power_matches_statsmodels():
    h = proportion_effectsize(prop1=0.12, prop2=0.10)
    expected = NormalIndPower().power(effect_size=h, nobs1=1000, alpha=0.05, ratio=1)
    table = stat.ab_power(baseline=0.10, lift=0.02, n_control=1000)
    assert value(table, 'power') == pytest.approx(expected, abs=1e-4)


def test_power_at_required_size_reaches_target():
    size = stat.ab_sample_size(baseline=0.10, lift=0.02)
    table = stat.ab_power(baseline=0.10, lift=0.02, n_control=value(size, 'n_control'))
    assert value(table, 'power') >= 0.8


# ab_mde()
@pytest.mark.parametrize('metric, kwargs', [
    ('proportion', {}),
    ('mean', {'sd': 25000}),
])
def test_mde_round_trips_to_target_power(metric, kwargs):
    baseline = 0.10 if metric == 'proportion' else 13000
    mde = stat.ab_mde(metric=metric, baseline=baseline, n_control=2000, **kwargs)
    table = stat.ab_power(
        metric=metric, baseline=baseline, lift=value(mde, 'mde'),
        n_control=2000, **kwargs,
    )
    assert value(table, 'power') == pytest.approx(0.8, abs=1e-3)


def test_mde_shrinks_with_more_customers():
    small = stat.ab_mde(baseline=0.10, n_control=1000)
    large = stat.ab_mde(baseline=0.10, n_control=4000)
    assert value(large, 'mde') < value(small, 'mde')


# n_treatment
@pytest.mark.parametrize('func, kwargs', [
    (stat.ab_power, {'lift': 0.02}),
    (stat.ab_mde, {}),
])
def test_n_treatment_equals_ratio(func, kwargs):
    by_count = func(baseline=0.10, n_control=4000, n_treatment=3850, **kwargs)
    by_ratio = func(baseline=0.10, n_control=4000, ratio=3850 / 4000, **kwargs)
    pd.testing.assert_frame_equal(by_count, by_ratio)


def test_n_treatment_is_reported_as_given():
    table = stat.ab_mde(baseline=0.10, n_control=4000, n_treatment=3850)
    assert value(table, 'n_treatment') == 3850


def test_power_with_n_treatment_matches_statsmodels():
    h = proportion_effectsize(prop1=0.12, prop2=0.10)
    expected = NormalIndPower().power(effect_size=h, nobs1=4000, alpha=0.05, ratio=3850 / 4000)
    table = stat.ab_power(baseline=0.10, lift=0.02, n_control=4000, n_treatment=3850)
    assert value(table, 'power') == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize('kwargs, message', [
    ({'n_treatment': 3850, 'ratio': 0.5}, '하나만'),
    ({'n_treatment': 1}, 'n_treatment'),
])
def test_n_treatment_rejects_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        stat.ab_mde(baseline=0.10, n_control=4000, **kwargs)


# 입력값 검사
@pytest.mark.parametrize('kwargs, message', [
    ({'metric': 'ratio', 'baseline': 0.1, 'lift': 0.02}, 'metric'),
    ({'baseline': 1.2, 'lift': 0.02}, 'baseline'),
    ({'baseline': 0.1}, 'lift'),
    ({'baseline': 0.95, 'lift': 0.1}, '처리군'),
    ({'metric': 'mean', 'baseline': 13000, 'lift': 1000}, 'sd'),
    ({'baseline': 0.1, 'lift': 0.02, 'icc': 0.01}, 'icc'),
    ({'baseline': 0.1, 'lift': 0.02, 'n_groups': 1}, 'n_groups'),
    ({'baseline': 0.1, 'lift': 0.02, 'alternative': 'smaller'}, 'alternative'),
])
def test_sample_size_rejects_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        stat.ab_sample_size(**kwargs)


def test_power_and_mde_require_n_control():
    with pytest.raises(ValueError, match='n_control'):
        stat.ab_power(baseline=0.1, lift=0.02)
    with pytest.raises(ValueError, match='n_control'):
        stat.ab_mde(baseline=0.1)
