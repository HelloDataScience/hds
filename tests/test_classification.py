# 분류 모델 평가 함수 테스트
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics

from hds import stat
from hds._utils import pos_proba, resolve_pos_label


# 0.3.5의 clf_cutoffs() 계산 방식(0과 1 범주 전용)
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


# clf_cutoffs()
def test_clf_cutoffs_matches_reference(clf_data):
    y, proba = clf_data
    result = stat.clf_cutoffs(y_true=y, y_prob=proba[:, 1])
    expected = reference_cutoffs(y_true=y, y_prob=proba[:, 1])

    for col in expected.columns:
        np.testing.assert_allclose(result[col], expected[col])

    np.testing.assert_allclose(result['TPR'], expected['Sensitivity'])
    np.testing.assert_allclose(result['FPR'], 1 - expected['Specificity'])


@pytest.mark.parametrize('kind', ['float', 'bool', 'str', 'string'])
def test_clf_cutoffs_supports_binary_labels(clf_data, kind):
    y, proba = clf_data
    expected = stat.clf_cutoffs(y_true=y, y_prob=proba[:, 1])
    labels = y.map({0: 'N', 1: 'Y'})

    if kind == 'float':
        y_true, pos_label = y.astype(float), None
    elif kind == 'bool':
        y_true, pos_label = y.astype(bool), None
    elif kind == 'str':
        y_true, pos_label = labels, 'Y'
    else:
        y_true, pos_label = labels.astype('string'), 'Y'

    result = stat.clf_cutoffs(y_true=y_true, y_prob=proba, pos_label=pos_label)
    pd.testing.assert_frame_equal(result, expected)


def test_clf_cutoffs_requires_pos_label_for_strings(clf_data):
    y, proba = clf_data
    with pytest.raises(ValueError, match='pos_label'):
        stat.clf_cutoffs(y_true=y.map({0: 'N', 1: 'Y'}), y_prob=proba)
