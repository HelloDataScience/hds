# 분류 모델 그래프 함수 테스트
import io
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
