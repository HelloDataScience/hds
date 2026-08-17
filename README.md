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
  KDE, 의사결정나무, 변수 중요도, 규제 회귀 계수 경로, ROC/PR 곡선,
  주성분·군집 진단 등)
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

### 선택 설치 옵션 (Extras)

기본 설치는 `numpy`·`pandas`·`scipy`·`matplotlib`·`seaborn`·`statsmodels`·
`scikit-learn`만 내려받습니다. 일부 함수는 아래 추가 패키지가 있어야 하며,
필요한 것만 골라 설치합니다.

| 옵션 | 추가 패키지 | 필요한 함수 |
| --- | --- | --- |
| `tree` | graphviz | `plot.tree()` |
| `font` | requests, beautifulsoup4 | `plot.add_google_font()` |
| `notebook` | ipywidgets, ipython | `stat.clf_metrics()`의 가로 배치 출력 |
| `varname` | varname | `plot.roc_curve()`·`plot.pr_curve()`의 범례 변수명 자동 표시 |
| `all` | 위 전체 | — |

```bash
pip install 'hds[tree]'   # 필요한 옵션만
pip install 'hds[all]'    # 0.2.x와 동일한 구성
```

설치하지 않은 상태로 해당 함수를 호출하면 설치 방법을 안내하는 메시지가
나타납니다. `stat.clf_metrics()`와 ROC·PR 곡선의 범례는 추가 패키지가 없어도
동작합니다.

> 의사결정나무 시각화 함수 `plot.tree()`는 파이썬 패키지 외에 시스템에도
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

> 시각화 함수는 `seaborn`과 마찬가지로 `plt.show()`를 호출하지 않고 `Axes`를
> 반환합니다. 주피터·VS Code 노트북은 셀 실행이 끝나면 그래프를 자동으로
> 출력하므로 위 코드만으로 충분하고, `.py` 스크립트에서는 마지막에
> `plt.show()`를 직접 호출하세요. 셀 마지막 줄에 찍히는 `<Axes: ...>` 문구는
> 코드 끝에 세미콜론(`;`)을 붙이면 사라집니다.

### 여러 그래프를 한 화면에 배치하기

`plot` 모듈의 시각화 함수는 `ax` 매개변수를 지원하고 그래프를 그린
matplotlib `Axes` 객체를 반환합니다. (PNG 파일로 저장하는 `plot.tree()`와
그래프 4종을 한 번에 그리는 `stat.regression_diagnosis()`는 제외입니다.)

- `ax`를 생략하면 현재 `Axes`에 그립니다. 함수는 `plt.show()`를 호출하지
  않으므로 그래프에 선이나 주석을 이어서 덧붙일 수 있습니다.
- `ax`를 지정하면 여러 함수를 하나의 `Figure`에 배치하거나 축·제목을 직접
  손볼 수 있습니다.

```python
# 함수가 그린 그래프에 그대로 이어서 덧그립니다.
plot.regline(data=cars, x='Weight', y='Price')
plt.axvline(x=1250, color='red', linestyle='--');
```

```python
import matplotlib.pyplot as plt
from hds import plot

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

plot.box_group(data=iris, x='species', y='petal_length', ax=axes[0, 0])
plot.regline(data=iris, x='petal_length', y='petal_width', ax=axes[0, 1])
plot.bar_freq(data=iris, x='species', ax=axes[1, 0])
plot.corr_heatmap(data=iris, ax=axes[1, 1])

fig.tight_layout()
plt.show()
```

ROC 곡선처럼 여러 모델을 겹쳐 그릴 때도 같은 `Axes`를 넘기고 `label`로
모델명을 지정합니다.

```python
fig, ax = plt.subplots()

plot.roc_curve(y_true=y_test, y_prob=prob_a, label='의사결정나무', ax=ax)
plot.roc_curve(y_true=y_test, y_prob=prob_b, label='랜덤 포레스트', ax=ax)
```

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
| `coef_path` | 규제 회귀(Lasso·Ridge·ElasticNet) 회귀계수 경로 |
| `roc_curve` / `pr_curve` | ROC 곡선·AUC / PR 곡선·AP |
| `roc_cutoff` | 최적 분류 기준점 시각화 |
| `screeplot` / `biplot` | 주성분 분석 진단 |
| `wcss` / `silhouette` | k-평균 군집 수 진단 |
| `add_google_font` | 구글 폰트 설치(한글 폰트 등) |

### `hds.stat` — 통계·진단

| 함수 | 설명 |
| --- | --- |
| `ols` / `glm` | 선형 회귀 / 로지스틱 회귀 적합 |
| `stepwise` | 변수선택법(`forward`·`backward`·`both`) |
| `regression_diagnosis` | 잔차 가정 진단 그래프 4종 |
| `vif` / `breusch_pagan` | 분산팽창지수 / 잔차 등분산성 검정 |
| `cooks_distance` / `leverage` / `augment` | 영향점·레버리지 진단 |
| `coefs` / `std_coefs` | 회귀계수 / 표준화 회귀계수 |
| `reg_metrics` / `clf_metrics` | 회귀 / 분류 성능 지표 |
| `clf_cutoffs` | 최적 분류 기준점 탐색(표) |

---

## 함수명 변경 (0.3.0)

`snake_case`로 이름을 통일하고, 그래프를 그리는 함수는 `plot` 모듈로
옮겼습니다. 이전 이름은 경고 없이 그대로 동작하므로 기존 코드를 고치지 않아도
됩니다. 새로 작성하는 코드에서는 새 이름을 사용하세요.

| 이전 이름 | 새 이름 |
| --- | --- |
| `stat.regmetrics` | `stat.reg_metrics` |
| `stat.clfmetrics` | `stat.clf_metrics` |
| `stat.breushpagan` | `stat.breusch_pagan` (철자 교정) |
| `stat.epi_roc` | `plot.roc_cutoff` |

---

## 대표 함수 시그니처 (API)

```python
# hds.plot  (tree를 제외한 시각화 함수는 ax=None을 받고 Axes를 반환)
box_group(data, x, y, palette=None, legend=False, ax=None) -> plt.Axes
scatter(data, x, y, color='0.3', ax=None) -> plt.Axes
regline(data, x, y, color='0.3', size=15, ax=None) -> plt.Axes
bar_freq(data, x, color=None, palette=None, legend=False, ax=None) -> plt.Axes
corr_heatmap(data, palette='RdYlBu', fontsize=8, ax=None) -> plt.Axes
kde2d(data, x, y, frac=0.2, seed=0, scatter=False, ax=None) -> plt.Axes
feature_importance(model, palette='Spectral', ax=None) -> plt.Axes
coef_path(X, y, model='lasso', alphas=None, l1_ratio=0.5, standardize=True,
          alpha=None, palette='Spectral', legend=True, ax=None) -> plt.Axes
roc_curve(y_true, y_prob, pos_label=None, color=None,
          label=None, ax=None) -> plt.Axes
roc_cutoff(y_true, y_prob, ax=None) -> plt.Axes
tree(model, file_name=None, class_name=None) -> None

# hds.stat
ols(y, X) -> statsmodels OLS
glm(y, X) -> statsmodels GLM
stepwise(y, X, direction='both') -> statsmodels OLS
regression_diagnosis(model) -> None
vif(model) -> pd.DataFrame
reg_metrics(y_true, y_pred) -> pd.DataFrame
clf_metrics(y_true, y_pred) -> None
clf_cutoffs(y_true, y_prob) -> pd.DataFrame
```

### 규제 회귀 계수 경로 예시

```python
from hds import plot

# alpha(규제 강도)가 커질수록 회귀계수가 0으로 수렴하는 과정을 확인
plot.coef_path(X=X_train, y=y_train, model='lasso')

# 교차검증으로 고른 alpha를 세로 점선으로 표시
plot.coef_path(X=X_train, y=y_train, model='lasso', alpha=model_cv.alpha_)
```

> 모든 함수는 한글 docstring을 제공합니다. `help(plot.box_group)` 또는
> `plot.box_group?`(Jupyter)로 매개변수 설명을 확인할 수 있습니다.

---

## 의존성 (Requirements)

- Python >= 3.10
- **필수**: numpy, pandas, scipy, matplotlib, seaborn(>=0.13),
  statsmodels, scikit-learn(>=1.4)
- **선택**: graphviz(`tree`), requests·beautifulsoup4(`font`),
  ipywidgets·ipython(`notebook`), varname(`varname`)

필수 패키지는 설치 시 자동으로 함께 설치되고, 선택 패키지는
[선택 설치 옵션](#선택-설치-옵션-extras)으로 필요할 때만 설치합니다.

---

## 라이선스 (License)

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

## 작성자 (Author)

**HelloDataScience** · [GitHub](https://github.com/HelloDataScience/hds) ·
hellodatasciencekorea@gmail.com
