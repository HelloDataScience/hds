# 관련 라이브러리 호출
import html
import keyword

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels
import statsmodels.api as sma
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oi
from scipy import stats
from sklearn import metrics
from statsmodels.regression.linear_model import RegressionModel

from hds._utils import (
    as_frame,
    pos_proba,
    resolve_pos_label,
)

# 이 모듈에서 공개하는 이름 목록
__all__ = [
    'ols',
    'forward_selection',
    'backward_selection',
    'stepwise_selection',
    'stepwise',
    'regression_diagnosis',
    'cooks_distance',
    'hat_matrix',
    'leverage',
    'std_resid',
    'augment',
    'breusch_pagan',
    'vif',
    'coefs',
    'std_coefs',
    'ols_table',
    'reg_metrics',
    'glm',
    'logit_table',
    'ClfMetrics',
    'clf_metrics',
    'cutoff_table',
    'epi_roc',
    'ab_sample_size',
    'ab_power',
    'ab_mde',
]


# 모듈에서 공개하는 이름만 반환하는 함수
def __dir__() -> list:
    """
    이 함수는 dir() 함수와 자동 완성 목록에 hds 함수만 보이도록 합니다.
    numpy, statsmodels처럼 이 모듈이 불러온 외부 이름은 목록에서
    제외합니다.

    반환값:
        공개하는 이름을 정렬한 리스트로 반환합니다.
    """
    return sorted(__all__)


# 입력변수 행렬의 복사본에 상수항을 추가하는 함수
def _add_const(X: pd.DataFrame, index: pd.Index = None) -> pd.DataFrame:
    """
    이 함수는 입력변수 행렬을 복사하고, 'const' 열이 없으면 첫 번째 열로
    상수항을 추가합니다. 원본 입력변수 행렬은 변경하지 않습니다.

    매개변수:
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
        index: X가 np.ndarray일 때 행 인덱스로 사용할 인덱스를 지정합니다.
            (기본값: None)

    반환:
        상수항을 추가한 입력변수 행렬의 복사본을 반환합니다.
    """
    X = as_frame(X=X, index=index)

    if 'const' not in X.columns:
        X.insert(loc=0, column='const', value=1)

    return X


# 선형 회귀 모델을 적합하는 함수
def ols(y: pd.Series, X: pd.DataFrame) -> statsmodels.api.OLS:
    """
    이 함수는 선형 회귀 모델을 적합합니다. 입력변수 행렬에 'const' 열이
    없으면 상수항을 추가하며, 원본 입력변수 행렬은 변경하지 않습니다.

    매개변수:
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
            np.ndarray의 열 이름은 x1, x2, ... 순서로 지정합니다.

    반환:
        선형 회귀 모델을 반환합니다.
    """
    X = _add_const(X=X, index=y.index if isinstance(y, pd.Series) else None)

    model = sma.OLS(endog=y, exog=X)

    return model.fit()


# 수식에 사용할 변수명을 반환하는 함수
def _term(name: str) -> str:
    """
    이 함수는 변수명을 statsmodels 수식에 사용할 수 있는 형태로 반환합니다.
    공백이나 특수문자가 있어 파이썬 식별자로 사용할 수 없는 변수명은 Q()로
    감쌉니다.

    매개변수:
        name: 변수명을 문자열로 지정합니다.

    반환:
        수식에 사용할 변수명을 문자열로 반환합니다.
    """
    if name.isidentifier() and not keyword.iskeyword(name):
        return name

    return f'Q({name!r})'


# 목표변수명과 입력변수명으로 선형 회귀 수식을 생성하는 함수
def _formula(y_name: str, x_vars: list) -> str:
    """
    이 함수는 목표변수명과 입력변수명으로 상수항을 포함하는 선형 회귀 수식을
    생성합니다. 입력변수가 없으면 상수항만 있는 수식을 생성합니다.

    매개변수:
        y_name: 목표변수명을 문자열로 지정합니다.
        x_vars: 입력변수명을 리스트로 지정합니다.

    반환:
        선형 회귀 수식을 문자열로 반환합니다.
    """
    terms = [_term(name=x_var) for x_var in x_vars] + ['1']

    return f'{_term(name=y_name)} ~ {" + ".join(terms)}'


# 변수선택법에 사용할 데이터를 준비하는 함수
def _selection_data(y: pd.Series, X: pd.DataFrame) -> tuple:
    """
    이 함수는 변수선택법에 사용할 목표변수와 입력변수를 하나의 데이터프레임으로
    합칩니다. 'const' 열은 제외하며, 원본 데이터는 변경하지 않습니다.

    매개변수:
        y: 목표변수 벡터를 pd.Series로 지정합니다. 이름이 없으면 'y'를
            사용합니다.
        X: 입력변수 행렬을 열 이름이 있는 pd.DataFrame으로 지정합니다.

    반환:
        합친 데이터프레임, 입력변수명 리스트, 목표변수명을 튜플로 반환합니다.
        입력변수명은 X의 열 순서를 유지합니다.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            '변수선택법의 X는 열 이름이 있는 pd.DataFrame으로 지정해야 합니다.'
        )

    if not isinstance(y, pd.Series):
        raise TypeError('변수선택법의 y는 pd.Series로 지정해야 합니다.')

    X = X.drop(columns='const', errors='ignore').rename(columns=str)
    y_name = 'y' if y.name is None else str(y.name)

    if y_name in X.columns:
        raise ValueError(
            f"목표변수명 '{y_name}'과 같은 이름의 열이 X에 있습니다."
        )

    data = pd.concat(objs=[X, y.rename(y_name)], axis=1)

    return data, list(X.columns), y_name


# 선형 회귀 모델을 전진선택법으로 적합하는 함수
def forward_selection(
    y: pd.Series,
    X: pd.DataFrame,
) -> statsmodels.formula.api.ols:
    """
    이 함수는 다중 선형 회귀 모델을 전진선택법으로 적합합니다.

    매개변수:
        y: 목표변수 벡터를 pd.Series로 지정합니다.
        X: 입력변수 행렬을 열 이름이 있는 pd.DataFrame으로 지정합니다.

    반환:
        전진선택법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    """
    data, x_vars, y_name = _selection_data(y=y, X=X)
    formula = _formula(y_name=y_name, x_vars=[])
    curr_aic = smf.ols(formula=formula, data=data).fit().aic

    selected = []

    while x_vars:
        aic_candidates = []
        for x_var in x_vars:
            formula = _formula(y_name=y_name, x_vars=selected + [x_var])
            aic = smf.ols(formula=formula, data=data).fit().aic
            aic = np.round(a=aic, decimals=4)
            aic_candidates.append((aic, x_var))

        aic_candidates.sort(reverse=True)
        new_aic, best_var = aic_candidates.pop()

        if curr_aic > new_aic:
            x_vars.remove(best_var)
            selected.append(best_var)
            curr_aic = new_aic
        else:
            break

    formula = _formula(y_name=y_name, x_vars=selected)
    model = smf.ols(formula=formula, data=data).fit()

    return model


# 선형 회귀 모델을 후진소거법으로 적합하는 함수
def backward_selection(
    y: pd.Series,
    X: pd.DataFrame,
) -> statsmodels.formula.api.ols:
    """
    이 함수는 선형 회귀 모델을 후진소거법으로 적합합니다.

    매개변수:
        y: 목표변수 벡터를 pd.Series로 지정합니다.
        X: 입력변수 행렬을 열 이름이 있는 pd.DataFrame으로 지정합니다.

    반환:
        후진소거법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    """
    data, x_vars, y_name = _selection_data(y=y, X=X)
    formula = _formula(y_name=y_name, x_vars=x_vars)
    curr_aic = smf.ols(formula=formula, data=data).fit().aic

    while x_vars:
        aic_candidates = []
        for x_var in x_vars:
            sub_vars = [var for var in x_vars if var != x_var]
            formula = _formula(y_name=y_name, x_vars=sub_vars)
            aic = smf.ols(formula=formula, data=data).fit().aic
            aic = np.round(a=aic, decimals=4)
            aic_candidates.append((aic, x_var))

        aic_candidates.sort(reverse=True)
        new_aic, best_var = aic_candidates.pop()

        if curr_aic > new_aic:
            x_vars.remove(best_var)
            curr_aic = new_aic
        else:
            break

    formula = _formula(y_name=y_name, x_vars=x_vars)
    model = smf.ols(formula=formula, data=data).fit()

    return model


# 선형 회귀 모델을 단계적방법으로 적합하는 함수
def stepwise_selection(
    y: pd.Series,
    X: pd.DataFrame,
) -> statsmodels.formula.api.ols:
    """
    이 함수는 선형 회귀 모델을 단계적방법으로 적합합니다.

    매개변수:
        y: 목표변수 벡터를 pd.Series로 지정합니다.
        X: 입력변수 행렬을 열 이름이 있는 pd.DataFrame으로 지정합니다.

    반환:
        단계적방법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    """
    data, x_vars, y_name = _selection_data(y=y, X=X)
    formula = _formula(y_name=y_name, x_vars=[])
    curr_aic = smf.ols(formula=formula, data=data).fit().aic

    selected = []

    while x_vars:
        aic_candidates = []
        for x_var in x_vars:
            formula = _formula(y_name=y_name, x_vars=selected + [x_var])
            aic = smf.ols(formula=formula, data=data).fit().aic
            aic = np.round(a=aic, decimals=4)
            aic_candidates.append((aic, 'add', x_var))

        if selected:
            for x_var in selected:
                sub_vars = [var for var in selected if var != x_var]
                formula = _formula(y_name=y_name, x_vars=sub_vars)
                aic = smf.ols(formula=formula, data=data).fit().aic
                aic = np.round(a=aic, decimals=4)
                aic_candidates.append((aic, 'sub', x_var))

        aic_candidates.sort(reverse=True)
        new_aic, how, best_var = aic_candidates.pop()

        if curr_aic > new_aic and how == 'add':
            x_vars.remove(best_var)
            selected.append(best_var)
            curr_aic = new_aic
        elif curr_aic > new_aic and how == 'sub':
            x_vars.append(best_var)
            selected.remove(best_var)
            curr_aic = new_aic
        elif curr_aic <= new_aic:
            break

    formula = _formula(y_name=y_name, x_vars=selected)
    model = smf.ols(formula=formula, data=data).fit()

    return model


# 선형 회귀 모델을 변수선택법으로 적합하는 함수
def stepwise(
    y: pd.Series,
    X: pd.DataFrame,
    direction: str = 'both',
) -> statsmodels.formula.api.ols:
    """
    이 함수는 세 가지 선형 회귀 모델의 변수선택법을 선택하는 함수입니다.

    매개변수:
        y: 목표변수 벡터를 pd.Series로 지정합니다.
        X: 입력변수 행렬을 열 이름이 있는 pd.DataFrame으로 지정합니다.
        direction: 변수선택법을 'forward', 'backward' 또는 'both'에서
            선택합니다.(기본값: 'both')

    반환:
        선택한 방법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    """
    if direction == 'forward':
        model = forward_selection(y, X)
    elif direction == 'backward':
        model = backward_selection(y, X)
    elif direction == 'both':
        model = stepwise_selection(y, X)
    else:
        raise ValueError(
            "direction은 'forward', 'backward', 'both' 중에서 지정해야 "
            f'합니다.(입력값: {direction!r})'
        )

    return model


# 선형 회귀 모델의 잔차진단 함수
def regression_diagnosis(model: statsmodels.api.OLS) -> tuple:
    """
    이 함수는 선형 회귀 모델의 잔차가정 만족 여부를 확인하는 다양한 그래프를
    그립니다. 정규 Q-Q 그래프와 Scale-Location 그래프의 표준화 잔차는
    레버리지를 반영한 내부 학생화 잔차이며, R의 plot() 함수와 같습니다.

    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을
            지정합니다.

    반환:
        네 가지 그래프를 그린 matplotlib Figure 객체와 2행 2열의 Axes 배열을
        튜플로 반환합니다. 주피터 노트북에서 반환값이 출력되지 않게 하려면
        fig, axes = stat.regression_diagnosis(model)처럼 변수에 할당하거나
        문장 끝에 세미콜론(;)을 붙입니다.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=100)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 선형성 가정
    # 잔차로 lowess(locally weighted linear regression) 회귀선을 산점도에 추가
    sns.regplot(
        x=model.fittedvalues,
        y=model.resid,
        lowess=True,
        scatter_kws=dict(color='0.8', ec='0.3', s=15),
        line_kws=dict(color='red', linewidth=1),
        ax=ax1,
    )

    ax1.axhline(y=0, color='0.5', linestyle='--', linewidth=1)

    ax1.set_title(
        label='Residuals vs Fitted',
        fontdict=dict(size=14, fontweight='bold'),
    )

    ax1.set_xlabel(xlabel='Fitted values', fontdict=dict(size=12))
    ax1.set_ylabel(ylabel='Residuals', fontdict=dict(size=12))

    # 정규성 가정 확인
    # 표준화 잔차(Standardized residuals): 레버리지를 반영한 내부 학생화 잔차
    stdres = np.asarray(model.get_influence().resid_studentized)

    # 이론상 분위수(Theoretical Quantiles)
    (x, y), _ = stats.probplot(x=stdres)

    # Q-Q plot
    sns.scatterplot(
        x=x,
        y=y,
        color='0.8',
        ec='0.3',
        size=2,
        legend=False,
        ax=ax2,
    )

    # 기준선(y = x)을 이론상 분위수 범위에 맞춰 추가
    lims = [x.min(), x.max()]
    ax2.plot(lims, lims, color='0.5', linestyle='--', linewidth=1)

    ax2.set_title(
        label='Normal Q-Q',
        fontdict=dict(size=14, fontweight='bold'),
    )

    ax2.set_xlabel(xlabel='Theoretical Quantiles', fontdict=dict(size=12))
    ax2.set_ylabel(ylabel='Standardized residuals', fontdict=dict(size=12))

    # 등분산성 가정 확인
    sns.regplot(
        x=model.fittedvalues,
        y=np.sqrt(np.abs(stdres)),
        lowess=True,
        scatter_kws=dict(color='0.8', ec='0.3', s=15),
        line_kws=dict(color='red', linewidth=1),
        ax=ax3,
    )

    ax3.set_title(
        label='Scale-Location',
        fontdict=dict(size=14, fontweight='bold'),
    )

    ax3.set_xlabel(xlabel='Fitted values', fontdict=dict(size=12))
    ax3.set_ylabel(
        ylabel='Sqrt of Standardized residuals',
        fontdict=dict(size=12),
    )

    # 쿡의 거리(이상치 탐지)
    sma.graphics.influence_plot(
        results=model,
        criterion='cooks',
        size=24,
        plot_alpha=0.2,
        ax=ax4,
    )

    for text in ax4.texts:
        text.set_fontsize(8)
        text.set_ha('center')
        text.set_va('center')

    fig.tight_layout()

    return fig, axes


# 쿡의 거리 계산 함수
def cooks_distance(model: statsmodels.api.OLS) -> pd.DataFrame:
    """
    이 함수는 선형 회귀 모델의 훈련셋으로 관측값별 쿡의 거리를 계산합니다.

    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을
            지정합니다.

    반환:
        훈련셋의 관측값별 쿡의 거리를 반환합니다.
    """
    cd, _ = oi.OLSInfluence(results=model).cooks_distance
    cd = cd.sort_values(ascending=False)

    return cd


# 햇 매트릭스 계산 함수
def hat_matrix(X: pd.DataFrame) -> np.ndarray:
    """
    이 함수는 입력변수 행렬로 햇 매트릭스(hat matrix)를 계산합니다. 입력변수
    행렬에 'const' 열이 없으면 상수항을 추가하며, 원본 입력변수 행렬은
    변경하지 않습니다.

    매개변수:
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.

    반환:
        훈련셋의 햇 매트릭스를 반환합니다. 행과 열의 개수가 관측값 개수와
        같으므로 관측값이 많으면 메모리를 많이 사용합니다.
    """
    X = np.asarray(_add_const(X=X), dtype=float)
    XtX = np.matmul(X.transpose(), X)
    XtX_inv = np.linalg.inv(XtX)
    result = np.matmul(np.matmul(X, XtX_inv), X.transpose())

    return result


# 레버리지(hat value) 계산 함수
def leverage(X: pd.DataFrame) -> pd.Series:
    """
    이 함수는 입력변수 행렬로 레버리지(hat value)를 계산합니다. 입력변수
    행렬에 'const' 열이 없으면 상수항을 추가하며, 원본 입력변수 행렬은
    변경하지 않습니다.

    매개변수:
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.

    반환:
        훈련셋의 관측값별 Leverage를 내림차순으로 정렬하여 반환합니다.
    """
    X = _add_const(X=X)
    values = np.asarray(X, dtype=float)

    # 햇 매트릭스 전체를 만들지 않고 대각 원소만 계산
    XtX_inv = np.linalg.inv(np.matmul(values.transpose(), values))
    hat = np.sum(np.matmul(values, XtX_inv) * values, axis=1)

    result = pd.Series(data=hat, index=X.index, name='Leverage')

    return result.sort_values(ascending=False)


# 표준화 잔차 계산 함수
def std_resid(model: statsmodels.api.OLS) -> pd.Series:
    """
    이 함수는 선형 회귀 모델의 잔차를 표준화합니다. 표준화 잔차는 잔차를
    잔차 표준오차와 sqrt(1 - 레버리지)의 곱으로 나눈 내부 학생화 잔차이며,
    R의 rstandard() 함수 및 augment() 함수의 std_resid 열과 같습니다.

    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을
            지정합니다.

    반환:
        훈련셋의 관측값별 표준화 잔차를 절대값의 내림차순으로 반환합니다.
        인덱스는 원래 관측값의 인덱스를 유지합니다.
    """
    resid = pd.Series(data=model.resid)

    stdres = pd.Series(
        data=np.asarray(model.get_influence().resid_studentized),
        index=resid.index,
    )

    return stdres.sort_values(ascending=False, key=lambda x: x.abs())


# 선형 회귀 모델의 영향점 계산 함수
def augment(model: statsmodels.api.OLS) -> pd.DataFrame:
    """
    이 함수는 선형 회귀 모델의 영향점을 계산합니다.

    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을
            지정합니다.

    반환:
        선형 회귀 모델의 영향점에 관련한 여러 지표를 데이터프레임으로
        반환합니다.
    """
    infl = model.get_influence()

    df1 = pd.DataFrame(
        data={model.model.endog_names: infl.endog},
        index=model.fittedvalues.index,
    )

    df2 = pd.DataFrame(
        data={
            'fitted': model.fittedvalues,
            'resid': model.resid,
            'hat': infl.hat_matrix_diag,
            'sigma': np.sqrt(infl.sigma2_not_obsi),
            'cooksd': infl.cooks_distance[0],
            'std_resid': infl.resid_studentized,
        }
    )

    result = pd.concat(objs=[df1, df2], axis=1)

    return result


# 잔차의 등분산성 검정 함수
def breusch_pagan(model: statsmodels.api.OLS) -> pd.DataFrame:
    """
    이 함수는 선형 회귀 모델의 잔차 등분산성 검정을 실행합니다.

    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을
            지정합니다.

    반환:
        선형 회귀 모델의 잔차 등분산성 검정 결과를 반환합니다.
    """
    test = sma.stats.het_breuschpagan(
        resid=model.resid,
        exog_het=model.model.exog,
    )

    result = pd.DataFrame(
        data=test,
        index=['Statistic', 'P-Value', 'F-Value', 'F P-Value'],
    ).T

    return result


# 분산팽창지수 반환 함수
def vif(model: statsmodels.api.OLS) -> pd.DataFrame:
    """
    이 함수는 입력변수 행렬의 분산팽창지수를 계산합니다. 모델에 상수항이
    있으면 상수항은 제외합니다.

    매개변수:
        model: statsmodels.formula.api 모듈 함수로 적합한 회귀 모델을
            지정합니다.

    반환:
        입력변수 행렬의 열별 분산팽창지수를 반환합니다.
    """
    func = oi.variance_inflation_factor
    exog = model.model.exog
    names = model.model.exog_names

    # 상수항은 열 이름 대신 statsmodels가 판별한 위치로 제외
    const_idx = model.model.data.const_idx
    indices = [i for i in range(len(names)) if i != const_idx]

    vifs = [func(exog=exog, exog_idx=i) for i in indices]
    result = pd.DataFrame(data=vifs, index=[names[i] for i in indices]).T

    return result


# 회귀계수 반환 함수
def coefs(model: object) -> pd.Series:
    """
    이 함수는 회귀 모델의 회귀계수를 확인합니다.

    매개변수:
        model: statsmodels로 적합한 회귀 모델 또는 coef_ 속성이 있는
            scikit-learn 모델을 지정합니다.

    반환:
        회귀 모델의 회귀계수를 pd.Series로 반환합니다. statsmodels 모델은
        상수항을 포함하고, scikit-learn 모델은 상수항(intercept_)을
        제외합니다. scikit-learn 모델의 회귀계수가 여러 행이면(다중 분류 등)
        행이 범주인 데이터프레임을 반환합니다.
    """
    # statsmodels 모델
    if hasattr(model, 'params'):
        params = model.params
        if isinstance(params, (pd.Series, pd.DataFrame)):
            return params.copy()
        return pd.Series(data=params, index=model.model.exog_names)

    # scikit-learn 모델
    if hasattr(model, 'coef_'):
        coef = np.asarray(model.coef_)
        names = getattr(model, 'feature_names_in_', None)
        if names is None:
            names = [f'x{i}' for i in range(coef.shape[-1])]

        if coef.ndim == 1:
            return pd.Series(data=coef, index=names)
        if coef.shape[0] == 1:
            return pd.Series(data=coef[0], index=names)

        return pd.DataFrame(
            data=coef,
            index=getattr(model, 'classes_', None),
            columns=names,
        )

    raise TypeError(
        f'{type(model).__name__} 모델은 지원하지 않습니다. statsmodels로 '
        '적합한 모델 또는 coef_ 속성이 있는 scikit-learn 모델을 지정하세요.'
    )


# 표준화된 회귀계수 반환 함수
def std_coefs(model: statsmodels.api.OLS) -> pd.Series:
    """
    이 함수는 회귀 모델의 표준화된 회귀계수를 계산합니다.

    매개변수:
        model: statsmodels의 OLS 또는 GLM으로 적합한 회귀 모델을 지정합니다.

    반환:
        회귀 모델의 표준화된 회귀계수를 반환합니다. 표준편차는 표본
        표준편차(ddof=1)로 계산하므로, OLS 모델은 입력변수와 목표변수를
        표준화하여 적합한 회귀계수와 같습니다. GLM 모델은 입력변수만
        표준화하므로 입력변수가 1 표준편차 증가할 때 선형 예측값(로지스틱
        회귀는 로그 오즈)의 변화량을 반환합니다.
    """
    fitted_model = getattr(model, 'model', None)

    if not isinstance(fitted_model, (sma.OLS, sma.GLM)):
        raise TypeError(
            f'{type(model).__name__} 모델은 지원하지 않습니다. '
            'statsmodels의 OLS 또는 GLM으로 적합한 모델을 지정하세요.'
        )

    X = pd.DataFrame(
        data=model.model.exog,
        columns=model.model.exog_names,
    )

    if isinstance(fitted_model, sma.OLS):
        # 목표변수도 입력변수와 같이 표본 표준편차(ddof=1)로 계산
        y = pd.Series(data=model.model.endog)
        result = model.params * (X.std() / y.std())
    else:
        result = model.params * (X.std() / 1)

    return result


# 회귀 모델의 회귀계수 검정 결과를 표로 정리하는 함수
def _coef_table(model: object, alpha: float) -> pd.DataFrame:
    """
    이 함수는 statsmodels로 적합한 회귀 모델의 회귀계수, 표준오차, 검정통계량,
    유의확률과 신뢰구간을 데이터프레임으로 정리합니다.

    매개변수:
        model: statsmodels로 적합한 회귀 모델을 지정합니다.
        alpha: 신뢰구간의 유의수준을 0과 1 사이의 실수로 지정합니다.

    반환:
        입력변수별 회귀계수 검정 결과를 데이터프레임으로 반환합니다.
        검정통계량 열 이름은 t 분포로 검정하면 't', 정규분포로 검정하면
        'z'입니다.
    """
    if not 0 < alpha < 1:
        raise ValueError(
            f'alpha는 0과 1 사이의 실수로 지정해야 합니다.(입력값: {alpha!r})'
        )

    statistic = 't' if model.use_t else 'z'
    ci = np.asarray(model.conf_int(alpha=alpha))

    result = pd.DataFrame(
        data={
            'coef': np.asarray(model.params),
            'std_err': np.asarray(model.bse),
            statistic: np.asarray(model.tvalues),
            'p_value': np.asarray(model.pvalues),
            'ci_lower': ci[:, 0],
            'ci_upper': ci[:, 1],
        },
        index=model.model.exog_names,
    )

    return result


# 선형 회귀 모델의 회귀계수 검정 결과표 반환 함수
def ols_table(
    model: statsmodels.api.OLS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    이 함수는 선형 회귀 모델의 회귀계수 검정 결과를 표로 정리합니다.

    매개변수:
        model: stat.ols() 또는 stat.stepwise() 함수처럼 statsmodels로 적합한
            선형 회귀 모델을 지정합니다.
        alpha: 신뢰구간의 유의수준을 지정합니다. 0.05이면 95% 신뢰구간을
            계산합니다.(기본값: 0.05)

    반환:
        입력변수별 회귀계수(coef), 표준오차(std_err), t 통계량(t),
        유의확률(p_value), 신뢰구간의 하한(ci_lower)과 상한(ci_upper)을
        데이터프레임으로 반환합니다. 강건한 표준오차(cov_type='HC3' 등)로
        적합한 모델은 정규분포로 검정하므로 t 대신 z 열을 반환합니다.
    """
    if not isinstance(getattr(model, 'model', None), RegressionModel):
        raise TypeError(
            f'{type(model).__name__} 모델은 지원하지 않습니다. statsmodels의 '
            'OLS 등으로 적합한 선형 회귀 모델을 지정하세요. 로지스틱 회귀 '
            '모델은 logit_table() 함수를 사용하세요.'
        )

    return _coef_table(model=model, alpha=alpha)


# 회귀 모델의 성능 지표 반환 함수
def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    이 함수는 회귀 모델의 다양한 성능 지표를 계산합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series, 1차원 np.ndarray 또는
            리스트로 지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series, 1차원 np.ndarray 또는
            리스트로 지정합니다.

    반환:
        회귀 모델의 다양한 성능 지표를 데이터프레임으로 반환합니다.
        실제값과 추정값 중 음수가 포함되면 MSLE와 RMSLE는 결측값으로
        처리합니다.
    """
    # 리스트나 object 자료형으로 지정해도 계산할 수 있도록 실수로 변환
    y_true = np.asarray(a=y_true, dtype=float)
    y_pred = np.asarray(a=y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            'y_true와 y_pred의 형태가 다릅니다.'
            f'({y_true.shape} != {y_pred.shape})'
        )

    R_2 = metrics.r2_score(y_true=y_true, y_pred=y_pred)
    MSE = metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
    RMSE = metrics.root_mean_squared_error(y_true=y_true, y_pred=y_pred)
    MAE = metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

    if (y_true < 0).any() or (y_pred < 0).any():
        MSLE = np.nan
        RMSLE = np.nan
    else:
        diff_log = np.log1p(y_true) - np.log1p(y_pred)
        MSLE = np.mean(diff_log ** 2)
        RMSLE = np.sqrt(MSLE)

    mask = (y_true != 0)
    if mask.any():
        MAPE = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        MAPE = metrics.mean_absolute_percentage_error(
            y_true=y_true,
            y_pred=y_pred,
        )

    result = pd.DataFrame(
        data={
            'metric': ['R^2', 'MSE', 'RMSE', 'MSLE', 'RMSLE', 'MAE', 'MAPE'],
            'score': [R_2, MSE, RMSE, MSLE, RMSLE, MAE, MAPE],
            'description': [
                'R Squared',
                'Mean Squared Error',
                'Root Mean Squared Error',
                'Mean Squared Log Error',
                'Root Mean Squared Log Error',
                'Mean Absolute Error',
                'Mean Absolute Percentage Error',
            ],
        }
    )

    return result


# 로지스틱 회귀 모델을 적합하는 함수
def glm(y: pd.Series, X: pd.DataFrame) -> statsmodels.api.GLM:
    """
    이 함수는 이항분포와 로짓 연결함수를 사용하는 GLM으로 로지스틱 회귀
    모델을 적합합니다. 입력변수 행렬에 'const' 열이 없으면 상수항을
    추가하며, 원본 입력변수 행렬은 변경하지 않습니다.

    매개변수:
        y: 목표변수 벡터를 0과 1로 이루어진 pd.Series 또는 1차원
            np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
            np.ndarray의 열 이름은 x1, x2, ... 순서로 지정합니다.

    반환:
        로지스틱 회귀 모델을 반환합니다.
    """
    X = _add_const(X=X, index=y.index if isinstance(y, pd.Series) else None)

    model = sma.GLM(endog=y, exog=X, family=sma.families.Binomial())

    return model.fit()


# 로지스틱 회귀 모델의 회귀계수 검정 결과와 오즈비 표 반환 함수
def logit_table(
    model: statsmodels.api.GLM,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    이 함수는 로지스틱 회귀 모델의 회귀계수 검정 결과와 오즈비를 표로
    정리합니다.

    매개변수:
        model: stat.glm() 함수로 적합한 로지스틱 회귀 모델 또는 statsmodels의
            Logit으로 적합한 모델을 지정합니다.
        alpha: 신뢰구간의 유의수준을 지정합니다. 0.05이면 95% 신뢰구간을
            계산합니다.(기본값: 0.05)

    반환:
        입력변수별 회귀계수(coef), 표준오차(std_err), z 통계량(z),
        유의확률(p_value), 오즈비(odds_ratio), 오즈비 신뢰구간의
        하한(or_ci_lower)과 상한(or_ci_upper)을 데이터프레임으로 반환합니다.
        오즈비는 입력변수가 1 증가할 때 오즈가 몇 배가 되는지를 나타내며,
        오즈비 신뢰구간에 1이 포함되면 해당 입력변수의 효과는 유의수준
        alpha에서 통계적으로 유의하지 않습니다.
    """
    fitted_model = getattr(model, 'model', None)

    is_logit = isinstance(fitted_model, sma.Logit)
    is_binomial_glm = (
        isinstance(fitted_model, sma.GLM)
        and isinstance(fitted_model.family, sma.families.Binomial)
        and isinstance(fitted_model.family.link, sma.families.links.Logit)
    )

    if not (is_logit or is_binomial_glm):
        raise TypeError(
            f'{type(model).__name__} 모델은 지원하지 않습니다. stat.glm() '
            '함수 또는 statsmodels의 Logit으로 적합한 로지스틱 회귀 모델을 '
            '지정하세요.'
        )

    table = _coef_table(model=model, alpha=alpha)

    # 회귀계수와 신뢰구간을 지수 변환하여 오즈비와 오즈비 신뢰구간을 계산
    result = table.drop(columns=['ci_lower', 'ci_upper'])
    result['odds_ratio'] = np.exp(table['coef'])
    result['or_ci_lower'] = np.exp(table['ci_lower'])
    result['or_ci_upper'] = np.exp(table['ci_upper'])

    return result


# 분류 모델의 성능 지표를 담는 클래스
class ClfMetrics:
    """
    이 클래스는 clf_metrics() 함수가 계산한 분류 모델의 성능 지표를 담습니다.
    주피터 노트북에서 셀의 마지막 줄로 실행하면 혼동행렬과 성능 지표를 가로로
    나란히 출력하고, print() 함수로 출력하면 세로로 출력합니다.

    속성:
        confusion_matrix: 혼동행렬을 담은 데이터프레임입니다.
        classification_report: 범주별 정밀도, 재현율, F1 점수, 도수와 정확도,
            평균 지표를 담은 데이터프레임입니다.
    """

    def __init__(
        self,
        confusion_matrix: pd.DataFrame,
        classification_report: pd.DataFrame,
        report_text: str,
    ) -> None:
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self._report_text = report_text

    # 콘솔에 출력할 문자열을 반환하는 메서드
    def __repr__(self) -> str:
        return (
            '▶ Confusion Matrix\n'
            f'{self.confusion_matrix.to_string()}\n\n'
            '▶ Classification Report\n'
            f'{self._report_text}'
        )

    # 주피터 노트북에 출력할 HTML을 반환하는 메서드
    def _repr_html_(self) -> str:
        title = '<pre style="margin: 0 0 4px 0;">{}</pre>'
        left = title.format('▶ Confusion Matrix')
        left += self.confusion_matrix.to_html()
        right = title.format('▶ Classification Report')
        right += f'<pre>{html.escape(self._report_text)}</pre>'

        return (
            '<div style="display: flex; align-items: flex-start; gap: 20px;">'
            f'<div>{left}</div><div>{right}</div>'
            '</div>'
        )


# 분류 모델의 성능 지표 반환 함수
def clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClfMetrics:
    """
    이 함수는 분류 모델의 다양한 성능 지표를 계산합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.

    반환:
        혼동행렬과 성능 지표를 담은 ClfMetrics 객체를 반환합니다. 주피터
        노트북에서 셀의 마지막 줄로 실행하면 혼동행렬과 성능 지표를 가로로
        나란히 출력합니다. 셀 중간이나 파이썬 스크립트에서는 print() 함수로
        출력합니다. 결과를 변수에 할당하면 confusion_matrix와
        classification_report 속성으로 데이터프레임을 사용할 수 있습니다.
    """
    y_labels = sorted(pd.Series(data=y_true).unique())
    cfm_labels = y_labels + ['All']
    cfm = pd.crosstab(index=y_true, columns=y_pred, margins=True)
    cfm = cfm.reindex(index=cfm_labels, columns=cfm_labels, fill_value=0)
    cfm.index = [f'True_{i}' for i in cfm_labels]
    cfm.columns = [f'Pred_{i}' for i in cfm_labels]

    report_text = metrics.classification_report(
        y_true=y_true,
        y_pred=y_pred,
        digits=4,
    )

    report_dict = metrics.classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True,
    )

    accuracy = report_dict.pop('accuracy', None)
    report = pd.DataFrame(data=report_dict).T
    report['support'] = report['support'].astype(int)

    # 정확도는 텍스트 보고서처럼 평균 지표 위에 F1 점수와 도수로 기록
    if accuracy is not None:
        is_avg = report.index.str.endswith(' avg')
        accuracy_row = pd.DataFrame(
            data={
                'precision': [np.nan],
                'recall': [np.nan],
                'f1-score': [accuracy],
                'support': [len(y_true)],
            },
            index=['accuracy'],
        )
        report = pd.concat(
            objs=[report[~is_avg], accuracy_row, report[is_avg]],
        )

    return ClfMetrics(
        confusion_matrix=cfm,
        classification_report=report,
        report_text=report_text,
    )


# 분류 모델의 분류 기준점별 성능 지표 표 반환 함수(TPR, FPR, MCC)
def cutoff_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: str | int = None,
) -> pd.DataFrame:
    """
    이 함수는 분류 기준점별 성능 지표를 표로 정리합니다. 최적의 분류 기준점을
    그래프로 확인하려면 plot.roc_cutoff() 함수를 사용하세요.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        y_prob: 목표변수의 예측 확률을 지정합니다. 양성 범주의 확률을 담은
            1차원 np.ndarray 또는 predict_proba() 함수가 반환한 2차원
            np.ndarray를 지정할 수 있습니다.
        pos_label: 양성 범주를 지정합니다. 생략하면 목표변수의 범주가 0과
            1이면 1, False와 True이면 True를 양성 범주로 사용하며, 그 밖의
            범주는 반드시 지정해야 합니다.(기본값: None)

    반환:
        분류 모델의 분류 기준점별로 TPR, FPR, MCC 등을 반환합니다.
    """
    pos_label = resolve_pos_label(y_true=y_true, pos_label=pos_label)
    y_prob = pos_proba(y_true=y_true, y_prob=y_prob, pos_label=pos_label)

    # 실제값을 양성 범주 여부로 변환
    actual = (
        pd.Series(data=y_true)
        .eq(other=pos_label)
        .fillna(value=False)
        .to_numpy(dtype=bool)
    )

    cutoffs = np.linspace(0, 1, 101)
    sens = []
    spec = []
    prec = []
    mccs = []

    for cutoff in cutoffs:
        pred = y_prob >= cutoff

        tp = np.sum(actual & pred)
        fn = np.sum(actual & ~pred)
        fp = np.sum(~actual & pred)
        tn = np.sum(~actual & ~pred)

        # 분모가 0이면 classification_report(zero_division=True)처럼 1로 처리
        sens.append(tp / (tp + fn) if tp + fn > 0 else 1.0)
        spec.append(tn / (tn + fp) if tn + fp > 0 else 1.0)
        prec.append(tp / (tp + fp) if tp + fp > 0 else 1.0)

        mcc = metrics.matthews_corrcoef(y_true=actual, y_pred=pred)
        mccs.append(mcc)

    result = pd.DataFrame(
        data={
            'Cutoff': cutoffs,
            'Sensitivity': sens,
            'Specificity': spec,
            'Precision': prec,
            'MCC': mccs,
        }
    )

    # The Optimal Point is the sum of Sensitivity and Specificity.
    result['Optimal'] = result['Sensitivity'] + result['Specificity']

    # TPR and FPR for ROC Curve.
    result['TPR'] = result['Sensitivity']
    result['FPR'] = 1 - result['Specificity']

    # Set column order.
    cols = [
        'Cutoff', 'Sensitivity', 'Specificity', 'Optimal',
        'Precision', 'TPR', 'FPR', 'MCC',
    ]

    # Select columns
    result = result[cols]

    return result


# 최적의 분류 기준점 시각화 함수(이전 이름)
def epi_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: plt.Axes = None,
    pos_label: str | int = None,
) -> plt.Axes:
    """
    이 함수는 'hds.plot.roc_cutoff' 함수의 이전 이름이며 동작이 같습니다.
    시각화 함수는 hds.plot 모듈로 옮겼으므로 새로 작성하는 코드에서는
    'plot.roc_cutoff' 함수를 사용하세요.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        y_prob: 목표변수의 예측 확률을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그립니다.(기본값: None)
        pos_label: 양성 범주를 지정합니다. 생략하면 목표변수의 범주가 0과
            1이면 1, False와 True이면 True를 양성 범주로 사용하며, 그 밖의
            범주는 반드시 지정해야 합니다.(기본값: None)

    반환:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    # 순환 참조를 피하려고 함수 안에서 호출
    from hds.plot import roc_cutoff

    return roc_cutoff(y_true=y_true, y_prob=y_prob, ax=ax, pos_label=pos_label)


# A/B 테스트 설계값을 확인하고 정리하는 함수
def _ab_setup(
    metric: str,
    baseline: float,
    sd: float,
    alpha: float,
    power: float,
    ratio: float,
    n_groups: int,
    icc: float,
    cluster_size: int,
    alternative: str,
) -> dict:
    """
    이 함수는 ab_sample_size(), ab_power(), ab_mde() 함수가 공통으로 쓰는
    입력값을 검사하고, 비교 1회당 유의수준과 설계효과를 계산합니다.
    """
    if metric not in ('proportion', 'mean'):
        raise ValueError("metric은 'proportion' 또는 'mean'으로 지정합니다.")

    if baseline is None:
        raise ValueError('baseline(대조군의 현재 수준)을 지정해야 합니다.')

    if metric == 'proportion' and not 0 < baseline < 1:
        raise ValueError('비율 지표의 baseline은 0과 1 사이로 지정합니다.')

    if metric == 'mean' and (sd is None or sd <= 0):
        raise ValueError('평균 지표는 sd(표준편차)를 양수로 지정해야 합니다.')

    if alternative not in ('two-sided', 'larger'):
        raise ValueError("alternative는 'two-sided' 또는 'larger'로 지정합니다.")

    if power is not None and not 0 < power < 1:
        raise ValueError('power는 0과 1 사이로 지정합니다.')

    if not 0 < alpha < 1:
        raise ValueError('alpha는 0과 1 사이로 지정합니다.')

    if ratio <= 0:
        raise ValueError('ratio는 양수로 지정합니다.')

    if n_groups < 2:
        raise ValueError('n_groups는 대조군을 포함해 2 이상으로 지정합니다.')

    if (icc is None) != (cluster_size is None):
        raise ValueError('점포 단위 실험은 icc와 cluster_size를 함께 지정합니다.')

    if icc is not None and not 0 <= icc <= 1:
        raise ValueError('icc는 0과 1 사이로 지정합니다.')

    if cluster_size is not None and cluster_size < 1:
        raise ValueError('cluster_size는 1 이상으로 지정합니다.')

    # 처리군이 여러 개이면 대조군과의 비교 횟수로 유의수준을 나눔(Bonferroni)
    alpha_adj = alpha / (n_groups - 1)

    # 같은 점포(군집) 고객끼리 비슷한 만큼 필요한 인원이 늘어남
    if icc is None:
        deff = 1.0
    else:
        deff = 1 + (cluster_size - 1) * icc

    return {'alpha_adj': alpha_adj, 'deff': deff}


# 처리군 인원을 대조군 대비 비율로 바꾸는 함수
def _ab_ratio(n_control: int, n_treatment: int, ratio: float) -> float:
    """
    이 함수는 ab_power(), ab_mde() 함수에서 n_treatment(처리군 인원)를 지정하면
    n_treatment ÷ n_control을 ratio로 반환하고, 지정하지 않으면 ratio를 그대로
    반환합니다.
    """
    if n_control is None or n_control < 2:
        raise ValueError('n_control은 2 이상으로 지정합니다.')

    if n_treatment is None:
        return ratio

    if ratio != 1.0:
        raise ValueError('n_treatment와 ratio는 둘 중 하나만 지정합니다.')

    if n_treatment < 2:
        raise ValueError('n_treatment는 2 이상으로 지정합니다.')

    return n_treatment / n_control


# 기대 개선폭을 절대 차이로 바꾸는 함수
def _ab_lift(baseline: float, lift: float, relative: bool) -> float:
    """
    이 함수는 relative=True로 지정한 상대 개선폭(예: 0.2 = 20%)을 baseline에
    곱해 절대 차이로 바꿉니다.
    """
    if lift is None or lift == 0:
        raise ValueError('lift(기대 개선폭)를 0이 아닌 값으로 지정해야 합니다.')

    return baseline * lift if relative else lift


# 지표별 효과 크기와 검정력 계산 객체를 반환하는 함수
def _ab_effect(
    metric: str,
    baseline: float,
    lift_abs: float,
    sd: float,
) -> tuple:
    """
    이 함수는 비율 지표는 Cohen's h와 정규분포 기반 검정력 계산 객체를,
    평균 지표는 Cohen's d와 t분포 기반 검정력 계산 객체를 반환합니다.
    """
    from statsmodels.stats.power import NormalIndPower, TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    if metric == 'proportion':
        treated = baseline + lift_abs
        if not 0 < treated < 1:
            raise ValueError(
                '처리군의 기대 비율(baseline + lift)이 0과 1 사이가 아닙니다.'
            )
        effect = proportion_effectsize(prop1=treated, prop2=baseline)
        return effect, NormalIndPower(), "Cohen's h"

    effect = lift_abs / sd
    return effect, TTestIndPower(), "Cohen's d"


# 비교 1회의 효과 크기를 기대 개선폭으로 되돌리는 함수
def _ab_effect_to_lift(
    metric: str,
    baseline: float,
    effect: float,
    sd: float,
) -> float:
    """
    이 함수는 ab_mde() 함수에서 구한 최소 효과 크기를 원래 단위의 절대 차이로
    바꿉니다.
    """
    if metric == 'proportion':
        phi = 2 * np.arcsin(np.sqrt(baseline)) + effect
        if phi > np.pi:
            return np.nan
        return np.sin(phi / 2) ** 2 - baseline

    return effect * sd


# 결과표를 만드는 함수
def _ab_table(rows: list) -> pd.DataFrame:
    """
    이 함수는 (항목, 값, 설명) 튜플의 리스트를 reg_metrics() 함수와 같은
    형태의 데이터프레임으로 변환합니다.
    """
    # 인원·점포 수·일수는 정수로, 나머지 실수는 소수점 넷째 자리까지 정리
    counts = ('n_', 'clusters_', 'days')
    tidy = []
    for item, value, desc in rows:
        if item.startswith(counts):
            value = int(np.ceil(value))
        elif isinstance(value, (float, np.floating)):
            value = round(float(value), 4)
        tidy.append((item, value, desc))

    return pd.DataFrame(
        data=tidy,
        columns=['item', 'value', 'description'],
        dtype=object,
    )


# A/B 테스트에 필요한 표본 크기를 계산하는 함수
def ab_sample_size(
    metric: str = 'proportion',
    baseline: float = None,
    lift: float = None,
    relative: bool = False,
    sd: float = None,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    n_groups: int = 2,
    icc: float = None,
    cluster_size: int = None,
    daily_traffic: float = None,
    alternative: str = 'two-sided',
) -> pd.DataFrame:
    """
    이 함수는 A/B 테스트에서 기대하는 효과를 잡아내는 데 필요한 그룹별 인원을
    계산합니다. 비율 지표(구매전환율, 쿠폰 사용률, 재구매율 등)는 Cohen's h와
    정규분포, 평균 지표(객단가, 고객당 매출 등)는 Cohen's d와 t분포를
    사용합니다.

    매개변수:
        metric: 평가 지표의 종류를 'proportion'(비율) 또는 'mean'(평균)으로
            지정합니다.(기본값: 'proportion')
        baseline: 대조군의 현재 수준을 지정합니다. 비율 지표는 0과 1 사이의
            비율(예: 0.10), 평균 지표는 평균(예: 52000)입니다.
        lift: 처리군에서 기대하는 개선폭을 지정합니다. relative=False이면
            절대 차이(예: 0.02 = 2%p, 2000 = 2,000원), relative=True이면
            baseline 대비 비율(예: 0.2 = 20%)입니다.
        relative: lift를 상대 개선폭으로 볼지 지정합니다.(기본값: False)
        sd: 평균 지표의 표준편차를 지정합니다. 과거 데이터에서 계산한 값을
            사용하며, 평균 지표는 반드시 지정해야 합니다. 두 그룹의
            표준편차가 같다고 가정하므로, 고객당 매출처럼 0원이 많은
            지표에서 구매 고객이 늘면 처리군의 표준편차가 더 커질 수
            있습니다. 이때는 sd를 10% 정도 크게 지정하세요.(기본값: None)
        alpha: 유의수준을 지정합니다.(기본값: 0.05)
        power: 검정력을 지정합니다.(기본값: 0.8)
        ratio: 대조군 대비 처리군 인원의 배수를 지정합니다. 처리군을 대조군의
            4배로 배정하면 4, 절반으로 배정하면 0.5입니다.(기본값: 1.0)
        n_groups: 대조군을 포함한 그룹 수를 지정합니다. 3 이상이면 처리군마다
            대조군과 비교하므로 유의수준을 비교 횟수로 나눕니다(Bonferroni).
            (기본값: 2)
        icc: 점포처럼 고객을 묶어서 배정할 때, 같은 묶음 안 고객끼리 비슷한
            정도(급내상관계수)를 지정합니다. cluster_size와 함께
            지정합니다.(기본값: None)
        cluster_size: 묶음(점포) 하나에 들어가는 평균 고객 수를
            지정합니다.(기본값: None)
        daily_traffic: 하루에 실험에 들어오는 전체 고객 수(모든 그룹
            합계)를 지정합니다. 지정하면 필요한 실험 일수를 함께
            계산합니다.(기본값: None)
        alternative: 'two-sided'(양측 검정) 또는 'larger'(처리군이 더 큰지만
            보는 단측 검정)를 지정합니다.(기본값: 'two-sided')

    반환:
        그룹별 필요 인원, 전체 인원, 효과 크기 등을 item, value, description
        열의 데이터프레임으로 반환합니다. 인원과 일수는 올림합니다.
    """
    setup = _ab_setup(
        metric=metric, baseline=baseline, sd=sd, alpha=alpha, power=power,
        ratio=ratio, n_groups=n_groups, icc=icc, cluster_size=cluster_size,
        alternative=alternative,
    )
    lift_abs = _ab_lift(baseline=baseline, lift=lift, relative=relative)
    effect, engine, effect_name = _ab_effect(
        metric=metric, baseline=baseline, lift_abs=lift_abs, sd=sd,
    )

    n_raw = engine.solve_power(
        effect_size=abs(effect),
        nobs1=None,
        alpha=setup['alpha_adj'],
        power=power,
        ratio=ratio,
        alternative=alternative,
    )

    # 군집 설계효과를 곱한 뒤 그룹별 인원을 올림
    n_control = int(np.ceil(n_raw * setup['deff']))
    n_treatment = int(np.ceil(n_raw * ratio * setup['deff']))
    n_total = n_control + n_treatment * (n_groups - 1)

    rows = [
        ('baseline', baseline, '대조군의 현재 수준'),
        ('treatment', baseline + lift_abs, '처리군의 기대 수준'),
        ('lift', lift_abs, '기대 개선폭(절대 차이)'),
        ('effect_size', effect, effect_name),
        ('alpha', setup['alpha_adj'], '비교 1회당 유의수준'),
        ('power', power, '검정력'),
        ('n_control', n_control, '대조군 인원'),
        ('n_treatment', n_treatment, '처리군 1개당 인원'),
        ('n_total', n_total, f'전체 인원({n_groups}개 그룹)'),
    ]

    if icc is not None:
        rows.append(('design_effect', setup['deff'], '설계효과 1 + (점포당 고객 수 - 1) × ICC'))
        rows.append((
            'clusters_control',
            int(np.ceil(n_control / cluster_size)),
            '대조군 점포(묶음) 수',
        ))
        rows.append((
            'clusters_treatment',
            int(np.ceil(n_treatment / cluster_size)),
            '처리군 1개당 점포(묶음) 수',
        ))

    if daily_traffic is not None:
        rows.append((
            'days',
            int(np.ceil(n_total / daily_traffic)),
            f'하루 {daily_traffic:,.0f}명 유입 시 필요 일수',
        ))

    return _ab_table(rows=rows)


# 주어진 인원으로 A/B 테스트의 검정력을 계산하는 함수
def ab_power(
    metric: str = 'proportion',
    baseline: float = None,
    lift: float = None,
    n_control: int = None,
    relative: bool = False,
    sd: float = None,
    alpha: float = 0.05,
    ratio: float = 1.0,
    n_groups: int = 2,
    icc: float = None,
    cluster_size: int = None,
    alternative: str = 'two-sided',
    n_treatment: int = None,
) -> pd.DataFrame:
    """
    이 함수는 이미 정해진 인원으로 A/B 테스트를 할 때, 기대하는 효과가 실제로
    있다면 그것을 잡아낼 확률(검정력)을 계산합니다. 매개변수는
    ab_sample_size() 함수와 같습니다.

    매개변수:
        n_control: 대조군 인원을 지정합니다. 처리군 인원은 n_control ×
            ratio입니다.
        n_treatment: 처리군 1개당 실제 인원을 지정합니다. 탈퇴나 발송 실패로
            처리군 인원이 대조군과 달라졌을 때 사용하며, 지정하면 ratio 대신
            n_treatment ÷ n_control을 사용합니다. ratio와 함께 지정할 수
            없습니다.
        그 밖의 매개변수는 ab_sample_size() 함수를 참고하세요.

    반환:
        검정력과 효과 크기 등을 item, value, description 열의 데이터프레임으로
        반환합니다.
    """
    ratio = _ab_ratio(n_control=n_control, n_treatment=n_treatment, ratio=ratio)
    setup = _ab_setup(
        metric=metric, baseline=baseline, sd=sd, alpha=alpha, power=None,
        ratio=ratio, n_groups=n_groups, icc=icc, cluster_size=cluster_size,
        alternative=alternative,
    )

    lift_abs = _ab_lift(baseline=baseline, lift=lift, relative=relative)
    effect, engine, effect_name = _ab_effect(
        metric=metric, baseline=baseline, lift_abs=lift_abs, sd=sd,
    )

    # 군집 설계효과만큼 실질 인원이 줄어듦
    n_eff = n_control / setup['deff']

    result = engine.power(
        effect_size=abs(effect),
        nobs1=n_eff,
        alpha=setup['alpha_adj'],
        ratio=ratio,
        alternative=alternative,
    )

    rows = [
        ('baseline', baseline, '대조군의 현재 수준'),
        ('treatment', baseline + lift_abs, '처리군의 기대 수준'),
        ('lift', lift_abs, '기대 개선폭(절대 차이)'),
        ('effect_size', effect, effect_name),
        ('alpha', setup['alpha_adj'], '비교 1회당 유의수준'),
        ('n_control', n_control, '대조군 인원'),
        ('n_treatment', n_treatment or n_control * ratio, '처리군 1개당 인원'),
        ('power', float(result), '검정력'),
    ]

    if icc is not None:
        rows.insert(-1, ('design_effect', setup['deff'], '설계효과 1 + (점포당 고객 수 - 1) × ICC'))

    return _ab_table(rows=rows)


# 주어진 인원으로 잡아낼 수 있는 최소 효과를 계산하는 함수
def ab_mde(
    metric: str = 'proportion',
    baseline: float = None,
    n_control: int = None,
    sd: float = None,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    n_groups: int = 2,
    icc: float = None,
    cluster_size: int = None,
    alternative: str = 'two-sided',
    n_treatment: int = None,
) -> pd.DataFrame:
    """
    이 함수는 이미 정해진 인원으로 A/B 테스트를 할 때, 주어진 검정력으로 잡아낼
    수 있는 최소 효과(MDE, Minimum Detectable Effect)를 계산합니다. 이보다
    작은 효과는 실제로 있어도 놓칠 가능성이 큽니다. 매개변수는
    ab_sample_size() 함수와 같습니다.

    매개변수:
        n_control: 대조군 인원을 지정합니다. 처리군 인원은 n_control ×
            ratio입니다.
        n_treatment: 처리군 1개당 실제 인원을 지정합니다. 탈퇴나 발송 실패로
            처리군 인원이 대조군과 달라졌을 때 사용하며, 지정하면 ratio 대신
            n_treatment ÷ n_control을 사용합니다. ratio와 함께 지정할 수
            없습니다.
        그 밖의 매개변수는 ab_sample_size() 함수를 참고하세요.

    반환:
        최소 효과의 절대 차이와 상대 차이 등을 item, value, description 열의
        데이터프레임으로 반환합니다.
    """
    ratio = _ab_ratio(n_control=n_control, n_treatment=n_treatment, ratio=ratio)
    setup = _ab_setup(
        metric=metric, baseline=baseline, sd=sd, alpha=alpha, power=power,
        ratio=ratio, n_groups=n_groups, icc=icc, cluster_size=cluster_size,
        alternative=alternative,
    )

    from statsmodels.stats.power import NormalIndPower, TTestIndPower

    if metric == 'proportion':
        engine, effect_name = NormalIndPower(), "Cohen's h"
    else:
        engine, effect_name = TTestIndPower(), "Cohen's d"

    n_eff = n_control / setup['deff']

    effect = engine.solve_power(
        effect_size=None,
        nobs1=n_eff,
        alpha=setup['alpha_adj'],
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    effect = float(effect)
    lift_abs = _ab_effect_to_lift(
        metric=metric, baseline=baseline, effect=effect, sd=sd,
    )

    rows = [
        ('baseline', baseline, '대조군의 현재 수준'),
        ('n_control', n_control, '대조군 인원'),
        ('n_treatment', n_treatment or n_control * ratio, '처리군 1개당 인원'),
        ('alpha', setup['alpha_adj'], '비교 1회당 유의수준'),
        ('power', power, '검정력'),
        ('effect_size', effect, effect_name),
        ('mde', lift_abs, '잡아낼 수 있는 최소 개선폭(절대 차이)'),
        ('mde_relative', lift_abs / baseline, '최소 개선폭(baseline 대비 비율)'),
        ('treatment', baseline + lift_abs, '처리군이 최소한 도달해야 하는 수준'),
    ]

    if icc is not None:
        rows.insert(3, ('design_effect', setup['deff'], '설계효과 1 + (점포당 고객 수 - 1) × ICC'))

    return _ab_table(rows=rows)


# End of Document
