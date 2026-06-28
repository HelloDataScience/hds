# hds

> Functions for EDA, Statistics and Machine Learning

[![PyPI version](https://img.shields.io/pypi/v/hds)](https://pypi.org/project/hds/)
[![Python](https://img.shields.io/pypi/pyversions/hds)](https://pypi.org/project/hds/)
[![License: MIT](https://img.shields.io/pypi/l/hds)](https://github.com/HelloDataScience/hds/blob/main/LICENSE)

`hds`는 **탐색적 데이터 분석(EDA)** 과정에서 자주 그리는 그래프와
통계·머신러닝 진단 작업을 한 줄로 끝낼 수 있도록 도와주는 파이썬
패키지입니다. `seaborn`/`matplotlib` 위에 얇게 얹어, 범주별 분포 비교·상관관계
히트맵·회귀 진단·ROC 곡선 등 **수업과 실무에서 반복되는 시각화를 함수
하나로** 제공합니다.

- 📊 **plot** — EDA 시각화 (상자그림, 산점도·회귀직선, 막대그래프, 히트맵,
  KDE, 의사결정나무, 변수 중요도, ROC/PR 곡선, 주성분·군집 진단 등)
- 📈 **stat** — 회귀 분석 도우미 (변수선택법, 잔차 진단, VIF, 영향점,
  표준화 회귀계수, 회귀·분류 성능 지표 등)

---

## 설치 (Installation)

```bash
pip install hds
```

이미 설치했다면 최신 버전으로 업그레이드합니다.

```bash
pip install --upgrade hds
```

> 의사결정나무 시각화 함수 `plot.tree()`는 시스템에
> [Graphviz](https://graphviz.org/download/) 실행 파일이 설치되어 있어야
> 합니다. (`brew install graphviz` 등)

---

## 빠른 시작 (Quick Start)

`seaborn`에 내장된 **iris** 데이터로 대표 그래프를 그려 봅니다.

```python
import seaborn as sns
from hds import plot

iris = sns.load_dataset('iris')

# 범주별 분포 비교 (상자그림 + 범주 평균 + 전체 평균선)
plot.box_group(data=iris, x='species', y='petal_length')

# 두 연속형 변수의 산점도 + 회귀직선
plot.regline(data=iris, x='petal_length', y='petal_width')

# 범주형 변수의 도수 막대그래프
plot.bar_freq(data=iris, x='species')

# 연속형 변수 간 상관계수 히트맵
plot.corr_heatmap(data=iris)

# 이차원 커널 밀도 추정(KDE) + 산점도
plot.kde2d(data=iris, x='petal_length', y='petal_width', scatter=True)
```

각 함수는 그래프를 그린 뒤 `plt.show()`를 호출하므로, 스크립트·노트북
어디서든 결과가 바로 표시됩니다.

### 회귀 분석 예시

```python
import seaborn as sns
from hds import stat

iris = sns.load_dataset('iris')
y = iris['petal_width']
X = iris[['petal_length', 'sepal_length', 'sepal_width']]

# 선형 회귀 적합 (상수항 자동 추가)
model = stat.ols(y=y, X=X)
print(model.summary())

stat.vif(model=model)             # 분산팽창지수(VIF)로 다중공선성 점검
stat.regression_diagnosis(model)  # 잔차 진단 그래프 4종
```

---

## 주요 기능 (Features)

### `hds.plot` — 시각화

| 함수 | 설명 |
| --- | --- |
| `box_group` | 범주별 상자그림 + 평균 비교 |
| `scatter` / `regline` | 산점도 / 산점도 + 회귀직선 |
| `bar_freq` | 범주형 변수 도수 막대그래프 |
| `bar_dodge_freq` / `bar_stack_freq` / `bar_stack_prop` | 소그룹 막대그래프(펼침·도수누적·비율누적) |
| `corr_heatmap` | 상관계수 히트맵 |
| `kde2d` | 이차원 커널 밀도(등고선) |
| `tree` | 의사결정나무 시각화(PNG 저장) |
| `feature_importance` | 입력변수 중요도 |
| `roc_curve` / `pr_curve` | ROC 곡선·AUC / PR 곡선·AP |
| `screeplot` / `biplot` | 주성분 분석 진단 |
| `wcss` / `silhouette` | k-평균 군집 수 진단 |
| `add_google_font` | 구글 폰트 설치(한글 폰트 등) |

### `hds.stat` — 통계·진단

| 함수 | 설명 |
| --- | --- |
| `ols` / `glm` | 선형 회귀 / 로지스틱 회귀 적합 |
| `stepwise` | 변수선택법(`forward`·`backward`·`both`) |
| `regression_diagnosis` | 잔차 가정 진단 그래프 4종 |
| `vif` | 분산팽창지수 |
| `cooks_distance` / `leverage` / `augment` | 영향점·레버리지 진단 |
| `coefs` / `std_coefs` | 회귀계수 / 표준화 회귀계수 |
| `regmetrics` / `clfmetrics` | 회귀 / 분류 성능 지표 |
| `clf_cutoffs` / `epi_roc` | 최적 분류 기준점 탐색·시각화 |

---

## 대표 함수 시그니처 (API)

```python
# hds.plot
box_group(data, x, y, palette=None, legend=False) -> None
scatter(data, x, y, color='0.3') -> None
regline(data, x, y, color='0.3', size=15) -> None
bar_freq(data, x, color=None, palette=None, legend=False) -> None
corr_heatmap(data, palette='RdYlBu', fontsize=8) -> None
kde2d(data, x, y, frac=0.2, seed=0, scatter=False) -> None
feature_importance(model, palette='Spectral') -> None
roc_curve(y_true, y_prob, pos_label=None, color=None) -> None
tree(model, file_name=None, class_name=None) -> None

# hds.stat
ols(y, X) -> statsmodels OLS
glm(y, X) -> statsmodels GLM
stepwise(y, X, direction='both') -> statsmodels OLS
regression_diagnosis(model) -> None
vif(model) -> pd.DataFrame
regmetrics(y_true, y_pred) -> pd.DataFrame
clf_cutoffs(y_true, y_prob) -> pd.DataFrame
```

> 모든 함수는 한글 docstring을 제공합니다. `help(plot.box_group)` 또는
> `plot.box_group?`(Jupyter)로 매개변수 설명을 확인할 수 있습니다.

---

## 의존성 (Requirements)

- Python >= 3.11
- numpy, pandas, scipy
- matplotlib, seaborn
- statsmodels, scikit-learn
- graphviz, requests, bs4, varname
- ipywidgets, ipython

설치 시 위 패키지가 자동으로 함께 설치됩니다.

---

## 라이선스 (License)

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

## 작성자 (Author)

**HelloDataScience** · [GitHub](https://github.com/HelloDataScience/hds) ·
hellodatasciencekorea@gmail.com
