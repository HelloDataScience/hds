# 분류 모델 그래프 함수 테스트
import pytest
from sklearn import metrics

from hds import plot


# 범례 문자열에서 AUC 또는 AP 값을 추출하는 함수
def legend_value(ax):
    text = ax.get_legend().get_texts()[0].get_text()
    return float(text.rsplit(':', maxsplit=1)[1])


def test_positive_class_is_majority(clf_data):
    y, _ = clf_data
    assert y.mean() > 0.5


@pytest.mark.parametrize('two_dim', [False, True])
def test_roc_curve_uses_class_1_by_default(clf_data, two_dim):
    y, proba = clf_data
    y_prob = proba if two_dim else proba[:, 1]
    ax = plot.roc_curve(y_true=y, y_prob=y_prob, label='model')
    expected = metrics.roc_auc_score(y_true=y, y_score=proba[:, 1])
    assert legend_value(ax) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize('two_dim', [False, True])
def test_pr_curve_uses_class_1_by_default(clf_data, two_dim):
    y, proba = clf_data
    y_prob = proba if two_dim else proba[:, 1]
    ax = plot.pr_curve(y_true=y, y_prob=y_prob, label='model')
    expected = metrics.average_precision_score(y_true=y, y_score=proba[:, 1])
    assert legend_value(ax) == pytest.approx(expected, abs=1e-4)


def test_pr_curve_with_string_labels(clf_data):
    y, proba = clf_data
    ax = plot.pr_curve(
        y_true=y.map({0: 'N', 1: 'Y'}),
        y_prob=proba,
        pos_label='Y',
        label='model',
    )
    expected = metrics.average_precision_score(y_true=y, y_score=proba[:, 1])
    assert legend_value(ax) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize('func', [plot.roc_curve, plot.pr_curve])
def test_string_labels_require_pos_label(clf_data, func):
    y, proba = clf_data
    with pytest.raises(ValueError, match='pos_label'):
        func(y_true=y.map({0: 'N', 1: 'Y'}), y_prob=proba, label='model')


def test_roc_cutoff_with_string_labels(clf_data):
    y, proba = clf_data
    ax = plot.roc_cutoff(
        y_true=y.map({0: 'N', 1: 'Y'}),
        y_prob=proba,
        pos_label='Y',
    )
    assert ax.get_title() == '최적의 분류 기준점 탐색'
