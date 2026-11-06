# 분류 모델 그래프 함수 테스트
import io
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
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


# urlopen 대신 반환할 가짜 응답 객체
class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


METADATA = b"""name: "Test Font"
fonts {
  style: "normal"
  weight: 100
  filename: "TestFont-Thin.ttf"
}
fonts {
  style: "normal"
  weight: 400
  filename: "TestFont-Regular.ttf"
}
fonts {
  style: "italic"
  weight: 400
  filename: "TestFont-Italic.ttf"
}
fonts {
  style: "normal"
  weight: 700
  filename: "TestFont-Bold.ttf"
}
"""


def test_google_font_urls_returns_all_files(monkeypatch):
    def fake_urlopen(url, timeout):
        if '/ofl/testfont/' not in url:
            raise urllib.error.HTTPError(url, 404, 'Not Found', None, None)
        return FakeResponse(METADATA)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    urls = plot._google_font_urls('Test Font')
    assert [url.rsplit('/', 1)[1] for url in urls] == [
        'TestFont-Thin.ttf',
        'TestFont-Regular.ttf',
        'TestFont-Italic.ttf',
        'TestFont-Bold.ttf',
    ]


def test_set_font_unknown_font(monkeypatch, tmp_path):
    def fake_urlopen(url, timeout):
        raise urllib.error.HTTPError(url, 404, 'Not Found', None, None)

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(matplotlib, 'get_cachedir', lambda: str(tmp_path))
    with pytest.raises(ValueError, match='fonts.google.com'):
        plot.set_font(font_name='No Such Font')
    assert not (tmp_path / 'hds_fonts').exists()


def test_set_font_downloads_once_and_sets_rcparams(monkeypatch, tmp_path):
    # matplotlib에 포함된 DejaVu Sans 파일을 구글 폰트 파일 대신 사용
    ttf = Path(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans.ttf')
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        if url.endswith('METADATA.pb'):
            return FakeResponse(METADATA)
        return FakeResponse(ttf.read_bytes())

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(matplotlib, 'get_cachedir', lambda: str(tmp_path))
    with matplotlib.rc_context():
        figsize = list(plt.rcParams['figure.figsize'])
        plot.set_font(font_name='Test Font')
        assert plt.rcParams['font.family'] == ['DejaVu Sans']
        assert plt.rcParams['font.size'] == 10
        assert plt.rcParams['axes.unicode_minus'] is False
        assert list(plt.rcParams['figure.figsize']) == figsize

        # 두 번째 실행에서는 내려받지 않음
        count = len(calls)
        plot.set_font(font_name='Test Font')
        assert len(calls) == count


# box_group 테스트용 데이터 (범주별 평균: a=1, b=2, c=3)
@pytest.fixture
def box_data():
    return pd.DataFrame(
        data={
            'group': ['b', 'b', 'a', 'a', 'c', 'c'],
            'value': [1.5, 2.5, 0.5, 1.5, 2.5, 3.5],
        }
    )


# x축 눈금 이름을 추출하는 함수
def tick_labels(ax):
    return [tick.get_text() for tick in ax.get_xticklabels()]


# 평균 숫자를 (x 위치, y 위치, 문자열)로 추출하는 함수
def mean_labels(ax):
    return [(text.get_position()[0], text.get_position()[1], text.get_text())
            for text in ax.texts]


def test_box_group_sorts_categories_by_default(box_data):
    ax = plot.box_group(data=box_data, x='group', y='value')
    assert tick_labels(ax) == ['a', 'b', 'c']
    assert mean_labels(ax) == [(0, 1.0, '1.00'), (1, 2.0, '2.00'), (2, 3.0, '3.00')]


def test_box_group_order_sets_ticks_and_mean_labels(box_data):
    ax = plot.box_group(data=box_data, x='group', y='value', order=['c', 'a', 'b'])
    assert tick_labels(ax) == ['c', 'a', 'b']
    assert mean_labels(ax) == [(0, 3.0, '3.00'), (1, 1.0, '1.00'), (2, 2.0, '2.00')]


def test_box_group_order_applies_palette_in_same_order(box_data):
    palette = ['#ff0000', '#00ff00', '#0000ff']
    ax = plot.box_group(data=box_data, x='group', y='value', palette=palette, order=['c', 'a', 'b'])
    boxes = [patch for patch in ax.patches if patch.get_facecolor()[3] > 0]
    # 각 상자의 x 위치 순서대로 색상을 확인합니다.
    boxes = sorted(boxes, key=lambda patch: patch.get_path().get_extents().x0)
    # seaborn은 상자 색의 채도를 기본값 0.75로 낮춰서 그립니다.
    expected = [matplotlib.colors.to_hex(sns.desaturate(color, 0.75)) for color in palette]
    assert [matplotlib.colors.to_hex(patch.get_facecolor()) for patch in boxes] == expected


def test_box_group_order_subset_keeps_overall_mean(box_data):
    ax = plot.box_group(data=box_data, x='group', y='value', order=['c', 'a'])
    assert tick_labels(ax) == ['c', 'a']
    assert ax.lines[-1].get_ydata()[0] == pytest.approx(2.0)


def test_box_group_order_rejects_unknown_category(box_data):
    with pytest.raises(ValueError, match='z'):
        plot.box_group(data=box_data, x='group', y='value', order=['a', 'z'])
