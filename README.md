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
| `notebook` | (없음) | 0.4.0부터 설치할 패키지 없음(이전 설치 명령 호환용) |
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

# 구글 폰트에서 확인한 이름으로 한글 폰트 설정
plot.set_font(font_name='Gowun Batang')

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

stat.ols_table(model=model)                   # 회귀계수 검정표(계수·표준오차·t·p·신뢰구간)
stat.vif(model=model)                         # 분산팽창지수(VIF)로 다중공선성 점검
fig, axes = stat.regression_diagnosis(model)  # 잔차 진단 그래프 4종
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
| `tree` | 의사결정나무 시각화(`image` 폴더에 PNG 저장) |
| `feature_importance` | 입력변수 중요도 |
| `coef_path` | 규제 회귀(Lasso·Ridge·ElasticNet) 회귀계수 경로 |
| `roc_curve` / `pr_curve` | ROC 곡선·AUC / PR 곡선·AP |
| `roc_cutoff` | 최적 분류 기준점 시각화 |
| `screeplot` / `biplot` | 주성분 분석 진단 |
| `wcss` / `silhouette` | k-평균 군집 수 진단 |
| `set_font` | 구글 폰트를 내려받아 그래프의 한글 폰트로 설정 |
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
| `ols_table` / `logit_table` | 회귀계수 검정표 / 회귀계수 검정표와 오즈비 |
| `reg_metrics` / `clf_metrics` | 회귀 / 분류 성능 지표 |
| `cutoff_table` | 분류 기준점별 성능 지표(표) |

---

## 변경 사항 (0.5.2)

- **`plot.set_font()`를 추가했습니다.** 구글 폰트에서 확인한 폰트명을
  지정하면 모든 굵기의 글꼴 파일을 matplotlib 임시 폴더에 내려받아 등록하고,
  그래프의 한글 폰트(글자 크기 10)와 `axes.unicode_minus=False`를 설정합니다.
  운영체제에 폰트를 설치하지 않으므로 관리자 권한이나 커널 재시작 없이
  바로 사용할 수 있고, 한 번 내려받은 파일은 다시 내려받지 않습니다.
  그래프 크기·해상도·범례는 바꾸지 않습니다.
  ```python
  plot.set_font(font_name='Gowun Batang')
  ```
- Noto Sans KR·Noto Serif KR처럼 구글 폰트 저장소에 가변 폰트 파일만 있는
  폰트는 matplotlib이 굵기를 고르지 못해 가는 글씨로 나옵니다.
- 기존 `plot.add_google_font()` 등 폰트 설치 함수는 그대로입니다.

---

## 변경 사항 (0.5.1)

- **`stat.clf_cutoffs()`의 이름을 `stat.cutoff_table()`로 바꿨습니다.**
  이름만 보고 `plot.roc_cutoff()`와 헷갈리기 쉬워서, 표를 반환하는 함수는
  `ols_table()`·`logit_table()`과 같은 `_table` 계열로 통일했습니다. 이전
  이름은 남기지 않았으므로 `stat.cutoff_table()`로 바꿔서 사용하세요.
  기준점을 그래프로 확인하는 `plot.roc_cutoff()`와 그 예전 이름
  `stat.epi_roc()`는 그대로입니다.
- `hds.plot`과 `hds.stat`에 `__all__`을 정의해 공개하는 이름을 명시했습니다.
  `dir(hds.stat)`과 자동 완성 목록에 hds 함수만 나타나며, 모듈이 내부에서
  불러온 `np`, `pd`, `plt`, `metrics`, `RegressionModel` 같은 이름은 목록에서
  빠집니다. 이름을 감췄을 뿐이므로 기존 코드의 동작은 그대로입니다.

---

## 변경 사항 (0.5.0)

- 0.3.0에서 남겨 두었던 이전 이름 `stat.regmetrics`, `stat.clfmetrics`,
  `stat.breushpagan`을 없앴습니다. 각각 `stat.reg_metrics`,
  `stat.clf_metrics`, `stat.breusch_pagan`으로 바꿔서 사용하세요.
  `stat.epi_roc`는 `plot.roc_cutoff`와 함께 그대로 둡니다.
- `stat.reg_metrics()`가 리스트와 object 자료형 입력도 실수로 변환해
  계산합니다. pandas 3에서 빈 `pd.Series()`에 예측값을 이어 붙이면 object
  자료형이 되는데, 이전에는 MSLE를 계산할 때 `TypeError`가 발생했습니다.
  실제값과 추정값의 길이가 다르면 `ValueError`가 발생합니다.

---

## 변경 사항 (0.4.3)

- `stat.regression_diagnosis()`의 정규 Q-Q 그래프에서 기준선(y = x)을 이론상
  분위수 범위에만 그립니다. 0.4.0~0.4.2에서는 표준화 잔차가 이론상 분위수보다
  멀리 떨어지면 기준선이 그만큼 길어져 가로축이 한쪽으로 늘어났습니다.

---

## 변경 사항 (0.4.2)

- `stat.regression_diagnosis()`의 정규 Q-Q 그래프와 Scale-Location 그래프,
  `stat.std_resid()`가 z-score 대신 내부 학생화 잔차(잔차 ÷ (잔차 표준오차 ×
  √(1 − 레버리지)))를 사용합니다. R의 `plot(lm)`·`rstandard()`와 같고
  `stat.augment()`의 `std_resid` 열과도 일치합니다. 레버리지가 큰 관측값의
  표준화 잔차는 이전보다 커지고, 나머지 관측값은 거의 그대로입니다.

---

## 변경 사항 (0.4.1)

- `stat.std_coefs()`가 OLS 모델의 표준화 회귀계수를 계산할 때 목표변수의
  표준편차도 입력변수와 같이 표본 표준편차(ddof=1)로 계산합니다. 이전에는
  목표변수만 모표준편차를 사용해 값이 √(n/(n−1))배 크게 나왔습니다. 이제
  입력변수와 목표변수를 표준화하여 적합한 회귀계수와 같습니다. GLM 모델의
  결과는 바뀌지 않습니다.

---

## 변경 사항 (0.4.0)

계산 오류를 바로잡고, 결과를 변수에 담아 다시 쓸 수 있도록 반환 형식을
정리했습니다. 기존 코드에서 달라질 수 있는 부분은 문자열 목표변수의
`pos_label` 지정, 파이썬 스크립트에서 `clf_metrics()` 결과의 `print()`,
`regression_diagnosis()` 반환값 출력입니다.

- **양성 범주 기본값을 바로잡았습니다.** `plot.roc_curve()`, `plot.pr_curve()`,
  `plot.roc_cutoff()`, `stat.clf_cutoffs()`는 `pos_label`을 생략하면 목표변수의
  범주가 0과 1이면 1, False와 True이면 True를 양성 범주로 사용합니다. 이전에는
  도수가 적은 범주를 양성으로 골라서, 1이 다수 범주인 데이터에서 AUC·AP가
  의도와 다르게 계산될 수 있었습니다. 문자열 범주는 양성 범주를 추측하지
  않으므로 `pos_label`을 지정해야 합니다.

  ```python
  plot.roc_curve(y_true=y_valid, y_prob=y_prob, pos_label='Pass')
  ```

- `stat.clf_cutoffs()`와 `plot.roc_cutoff()`가 0과 1 외의 이진 범주(실수,
  불리언, 문자열)와 `predict_proba()`가 반환한 2차원 확률을 받습니다.
- `stat.ols()`, `stat.glm()`, `stat.hat_matrix()`, `stat.leverage()`가 원본
  `X`에 `const` 열을 추가하지 않습니다. 2차원 `np.ndarray`도 받으며, 열 이름은
  `x1`, `x2`, ... 순서로 지정합니다.
- `stat.coefs()`가 statsmodels 모델(상수항 포함)과 scikit-learn 모델을 모두
  받습니다. 지원하지 않는 모델을 지정하면 `TypeError`가 발생합니다.
- `stat.std_resid()`가 원래 관측값의 인덱스를 유지합니다.
- `stat.vif()`가 상수항을 열 이름이 아닌 실제 위치로 판별하므로, 상수항이 없는
  모델에서 첫 번째 입력변수가 빠지지 않습니다.
- `stat.std_coefs()`는 OLS·GLM이 아닌 모델에 `TypeError`를, `stat.stepwise()`는
  잘못된 `direction`에 `ValueError`를 발생시킵니다.
- 변수선택법의 결과가 실행할 때마다 같은 변수 순서로 나오고, 공백이 있는 열
  이름도 처리합니다.
- `stat.clf_metrics()`가 혼동행렬과 성능 지표를 담은 결과 객체를 반환합니다.
  주피터 노트북에서 셀의 마지막 줄로 실행하면 이전처럼 한 번 출력하며,
  ipywidgets가 없어도 혼동행렬과 성능 지표를 가로로 나란히 배치합니다. 셀
  중간이나 파이썬 스크립트에서는 `print()`로 출력해야 합니다.

  ```python
  result = stat.clf_metrics(y_true=y_valid, y_pred=y_pred)
  result.confusion_matrix        # 혼동행렬(데이터프레임)
  result.classification_report   # 범주별 정밀도·재현율·F1 점수(데이터프레임)
  print(result)                  # 콘솔 출력
  ```

- `notebook` 추가 설치 옵션은 설치할 패키지가 없습니다. 이전 설치 명령이
  깨지지 않도록 옵션 이름만 남겨 두었습니다.
- `stat.regression_diagnosis()`가 그래프를 그린 `Figure`와 `Axes` 배열을
  반환하므로 그래프를 저장하거나 제목을 고칠 수 있습니다. 셀의 마지막 줄에서
  호출하면 그래프 아래에 반환값이 텍스트로 출력되므로 변수에 할당하거나 끝에
  `;`를 붙이세요. 정규 Q-Q 그래프의 기준선은 데이터 범위에 맞춰 그립니다.

  ```python
  fig, axes = stat.regression_diagnosis(model)
  fig.savefig('diagnosis.png')
  ```

- 회귀계수 검정 결과를 표로 정리하는 `stat.ols_table()`과 `stat.logit_table()`을
  추가했습니다. `logit_table()`은 `stat.glm()`으로 적합한 모델을 받아 오즈비와
  오즈비 신뢰구간을 함께 계산합니다. `alpha`로 신뢰구간의 유의수준을 바꿀 수
  있습니다.

  ```python
  stat.ols_table(stat.ols(y=y, X=X))
  # coef, std_err, t, p_value, ci_lower, ci_upper

  stat.logit_table(stat.glm(y=y, X=X))
  # coef, std_err, z, p_value, odds_ratio, or_ci_lower, or_ci_upper
  ```

- Python 3.11 이상이 필요합니다. Python 3.10에서는 `pip install hds`가
  0.3.5를 설치합니다.

---

## 변경 사항 (0.3.5)

- `plot.tree()`가 png 파일 외에는 어떤 파일도 만들지 않습니다. 0.3.4까지는
  Graphviz에 넘길 dot 소스를 확장자 없는 파일(예: `dt_base`)로 잠깐 썼다가
  지웠는데, 이제 소스를 파일로 쓰지 않고 Graphviz 프로세스에 직접 전달한 뒤
  결과 png만 저장합니다.

---

## 변경 사항 (0.3.3)

- `plot.tree()`가 PNG 파일을 **`image` 폴더에 저장**합니다. 현재 작업 경로가
  `image` 폴더이면 그 자리에, 그 밖에는 현재 작업 경로와 형제 관계인 `image`
  폴더에 저장하며 폴더가 없으면 새로 만듭니다. `path` 매개변수로 저장할
  폴더를 직접 지정할 수도 있습니다.

  ```python
  plot.tree(model)                      # 작업 경로가 data면 ../image/model.png
  plot.tree(model, path='../output')    # ../output/model.png
  ```

  중간 산출물인 `.dot` 파일을 아예 만들지 않으므로 작업 폴더에 임시 파일이
  남지 않습니다. 의사결정나무가 아닌 모델을 지정하면 안내 메시지와 함께
  `TypeError`가 발생합니다.
- `plot.biplot()`의 제목이 `x`·`y`로 지정한 주성분 번호를 따라갑니다. 이전에는
  `x=3, y=4`로 바꿔도 제목이 늘 `Biplot with PC1 and PC2`로 나와 축 이름과
  어긋났습니다. 변수 화살표도 주성분 개수가 아닌 변수 개수만큼 그리므로,
  주성분을 줄여서 계산한 결과에서 화살표가 빠지지 않습니다.
- `plot.bar_freq()`의 제목이 `x`로 지정한 변수명을 따라갑니다. 이전에는
  "목표변수의 범주별 도수 비교"로 고정되어 있었습니다.

---

## 함수명 변경 (0.3.0)

`snake_case`로 이름을 통일하고, 그래프를 그리는 함수는 `plot` 모듈로
옮겼습니다. 0.4.3까지는 이전 이름도 경고 없이 동작했지만, 0.5.0에서 아래 표의
이전 이름 세 개를 없앴습니다. `stat.epi_roc`는 그대로 사용할 수 있습니다.

| 이전 이름 | 새 이름 | 0.5.0 |
| --- | --- | --- |
| `stat.regmetrics` | `stat.reg_metrics` | 제거 |
| `stat.clfmetrics` | `stat.clf_metrics` | 제거 |
| `stat.breushpagan` | `stat.breusch_pagan` (철자 교정) | 제거 |
| `stat.epi_roc` | `plot.roc_cutoff` | 유지 |

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
roc_cutoff(y_true, y_prob, ax=None, pos_label=None) -> plt.Axes
tree(model, file_name=None, class_name=None, path=None) -> None

# hds.stat
ols(y, X) -> statsmodels OLS
glm(y, X) -> statsmodels GLM
ols_table(model, alpha=0.05) -> pd.DataFrame
logit_table(model, alpha=0.05) -> pd.DataFrame
stepwise(y, X, direction='both') -> statsmodels OLS
regression_diagnosis(model) -> (plt.Figure, np.ndarray)
vif(model) -> pd.DataFrame
reg_metrics(y_true, y_pred) -> pd.DataFrame
clf_metrics(y_true, y_pred) -> ClfMetrics
cutoff_table(y_true, y_prob, pos_label=None) -> pd.DataFrame
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

- Python >= 3.11
- **필수**: numpy, pandas, scipy, matplotlib, seaborn(>=0.13),
  statsmodels, scikit-learn(>=1.4)
- **선택**: graphviz(`tree`), requests·beautifulsoup4(`font`),
  varname(`varname`)

필수 패키지는 설치 시 자동으로 함께 설치되고, 선택 패키지는
[선택 설치 옵션](#선택-설치-옵션-extras)으로 필요할 때만 설치합니다.

---

## 라이선스 (License)

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

## 작성자 (Author)

**HelloDataScience** · [GitHub](https://github.com/HelloDataScience/hds) ·
hellodatasciencekorea@gmail.com
