# 분류 모델 평가 함수 테스트
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics

from hds import stat
from hds._utils import pos_proba, resolve_pos_label


# 0.3.5의 cutoff_table() 계산 방식(0과 1 범주 전용)
def reference_cutoffs(y_true, y_prob):
    rows = []
    for cutoff in np.linspace(0, 1, 101):
        pred = np.where(y_prob >= cutoff, 1, 0)
        report = metrics.classification_report(
            y_true=y_true,
            y_pred=pred,
            output_dict=True,
            zero_division=True,
        )
        rows.append(
            {
                'Sensitivity': report['1']['recall'],
                'Specificity': report['0']['recall'],
                'Precision': report['1']['precision'],
                'MCC': metrics.matthews_corrcoef(y_true=y_true, y_pred=pred),
            }
        )
    return pd.DataFrame(data=rows)


# resolve_pos_label(), pos_proba()
@pytest.mark.parametrize(
    ('y_true', 'expected'),
    [
        ([0, 1, 1], 1),
        ([0.0, 1.0, 1.0], 1),
        ([False, True, True], True),
    ],
)
def test_resolve_pos_label_default(y_true, expected):
    result = resolve_pos_label(y_true=np.array(y_true))
    assert result == expected
    assert type(result) is type(expected)


def test_resolve_pos_label_requires_label_for_strings():
    with pytest.raises(ValueError, match='pos_label'):
        resolve_pos_label(y_true=pd.Series(data=['N', 'Y', 'Y']))


def test_resolve_pos_label_rejects_unknown_label():
    with pytest.raises(ValueError):
        resolve_pos_label(y_true=pd.Series(data=['N', 'Y']), pos_label='yes')


def test_pos_proba_selects_positive_column(clf_data):
    y, proba = clf_data
    result_1 = pos_proba(y_true=y, y_prob=proba, pos_label=1)
    result_0 = pos_proba(y_true=y, y_prob=proba, pos_label=0)
    np.testing.assert_array_equal(result_1, proba[:, 1])
    np.testing.assert_array_equal(result_0, proba[:, 0])


def test_pos_proba_rejects_column_mismatch(clf_data):
    y, proba = clf_data
    extra = np.column_stack([proba, proba[:, 0]])
    with pytest.raises(ValueError):
        pos_proba(y_true=y, y_prob=extra, pos_label=1)


# cutoff_table()
def test_cutoff_table_matches_reference(clf_data):
    y, proba = clf_data
    result = stat.cutoff_table(y_true=y, y_prob=proba[:, 1])
    expected = reference_cutoffs(y_true=y, y_prob=proba[:, 1])

    for col in expected.columns:
        np.testing.assert_allclose(result[col], expected[col])

    np.testing.assert_allclose(result['TPR'], expected['Sensitivity'])
    np.testing.assert_allclose(result['FPR'], 1 - expected['Specificity'])


@pytest.mark.parametrize('kind', ['float', 'bool', 'str', 'string'])
def test_cutoff_table_supports_binary_labels(clf_data, kind):
    y, proba = clf_data
    expected = stat.cutoff_table(y_true=y, y_prob=proba[:, 1])
    labels = y.map({0: 'N', 1: 'Y'})

    if kind == 'float':
        y_true, pos_label = y.astype(float), None
    elif kind == 'bool':
        y_true, pos_label = y.astype(bool), None
    elif kind == 'str':
        y_true, pos_label = labels, 'Y'
    else:
        y_true, pos_label = labels.astype('string'), 'Y'

    result = stat.cutoff_table(
        y_true=y_true,
        y_prob=proba,
        pos_label=pos_label,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_cutoff_table_requires_pos_label_for_strings(clf_data):
    y, proba = clf_data
    with pytest.raises(ValueError, match='pos_label'):
        stat.cutoff_table(y_true=y.map({0: 'N', 1: 'Y'}), y_prob=proba)


# clf_metrics()
def test_clf_metrics_returns_result_without_printing(clf_data, capsys):
    y, proba = clf_data
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    result = stat.clf_metrics(y_true=y, y_pred=y_pred)
    assert isinstance(result, stat.ClfMetrics)
    assert capsys.readouterr().out == ''


def test_clf_metrics_confusion_matrix(clf_data):
    y, proba = clf_data
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    cfm = stat.clf_metrics(y_true=y, y_pred=y_pred).confusion_matrix
    expected = metrics.confusion_matrix(y_true=y, y_pred=y_pred)

    assert list(cfm.index) == ['True_0', 'True_1', 'True_All']
    assert list(cfm.columns) == ['Pred_0', 'Pred_1', 'Pred_All']
    np.testing.assert_array_equal(cfm.iloc[:2, :2], expected)
    assert cfm.loc['True_All', 'Pred_All'] == len(y)


def test_clf_metrics_classification_report(clf_data):
    y, proba = clf_data
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    report = stat.clf_metrics(y_true=y, y_pred=y_pred).classification_report
    expected = metrics.classification_report(
        y_true=y,
        y_pred=y_pred,
        output_dict=True,
    )

    rows = ['0', '1', 'accuracy', 'macro avg', 'weighted avg']
    assert list(report.index) == rows

    for row in ['0', '1', 'macro avg', 'weighted avg']:
        for col in ['precision', 'recall', 'f1-score', 'support']:
            assert report.loc[row, col] == pytest.approx(expected[row][col])

    assert report.loc['accuracy', 'f1-score'] == pytest.approx(
        expected['accuracy']
    )
    assert report.loc['accuracy', 'support'] == len(y)


def test_clf_metrics_display(clf_data):
    y, proba = clf_data
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    result = stat.clf_metrics(y_true=y, y_pred=y_pred)

    text = repr(result)
    assert text.startswith('▶ Confusion Matrix')
    assert '▶ Classification Report' in text

    html_text = result._repr_html_()
    assert html_text.count('<table') == 1
    assert '▶ Classification Report' in html_text


def test_clf_metrics_with_string_labels(clf_data):
    y, proba = clf_data
    labels = y.map({0: 'N', 1: 'Y'})
    y_pred = pd.Series(data=np.where(proba[:, 1] >= 0.5, 'Y', 'N'))
    result = stat.clf_metrics(y_true=labels, y_pred=y_pred)
    expected = ['True_N', 'True_Y', 'True_All']
    assert list(result.confusion_matrix.index) == expected


# 0.5.0과 0.5.1에서 없앤 이전 이름과 그대로 두는 이름
@pytest.mark.parametrize(
    'name',
    ['regmetrics', 'clfmetrics', 'breushpagan', 'clf_cutoffs'],
)
def test_old_names_are_removed(name):
    assert not hasattr(stat, name)


def test_epi_roc_is_kept(clf_data):
    y, proba = clf_data
    ax = stat.epi_roc(y_true=y, y_prob=proba)
    assert ax.get_title() == '최적의 분류 기준점 탐색'
