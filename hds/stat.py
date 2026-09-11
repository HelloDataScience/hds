# 관련 라이브러리 호출
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

from hds._utils import (
    as_frame,
    pos_proba,
    renamed_alias,
    resolve_pos_label,
    try_import,
)


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
def regression_diagnosis(model: statsmodels.api.OLS) -> None:
    """
    이 함수는 선형 회귀 모델의 잔차가정 만족 여부를 확인하는 다양한 그래프를
    그립니다.

    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을
            지정합니다.

    반환:
        네 가지 그래프를 하나의 Figure에 그리며, 반환하는 객체는 없습니다.
    """
    plt.figure(figsize=(10, 10), dpi=100)

    # 선형성 가정
    # 잔차로 lowess(locally weighted linear regression) 회귀선을 산점도에 추가
    ax1 = plt.subplot(2, 2, 1)

    sns.regplot(
        x=model.fittedvalues,
        y=model.resid,
        lowess=True,
        scatter_kws=dict(color='0.8', ec='0.3', s=15),
        line_kws=dict(color='red', linewidth=1),
        ax=ax1,
    )

    plt.axhline(y=0, color='0.5', linestyle='--', linewidth=1)

    plt.title(
        label='Residuals vs Fitted',
        fontdict=dict(size=14, fontweight='bold'),
    )

    plt.xlabel(xlabel='Fitted values', fontdict=dict(size=12))
    plt.ylabel(ylabel='Residuals', fontdict=dict(size=12))

    # 정규성 가정 확인
    ax2 = plt.subplot(2, 2, 2)

    # 표준화 잔차(Standardized residuals)
    stdres = pd.Series(data=stats.zscore(a=model.resid))

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

    plt.plot([-4, 4], [-4, 4], color='0.5', linestyle='--', linewidth=1)

    plt.title(
        label='Normal Q-Q',
        fontdict=dict(size=14, fontweight='bold'),
    )

    plt.xlabel(xlabel='Theoretical Quantiles', fontdict=dict(size=12))
    plt.ylabel(ylabel='Standardized residuals', fontdict=dict(size=12))

    # 등분산성 가정 확인
    ax3 = plt.subplot(2, 2, 3)

    sns.regplot(
        x=model.fittedvalues,
        y=np.sqrt(stdres.abs()),
        lowess=True,
        scatter_kws=dict(color='0.8', ec='0.3', s=15),
        line_kws=dict(color='red', linewidth=1),
        ax=ax3,
    )

    plt.title(
        label='Scale-Location',
        fontdict=dict(size=14, fontweight='bold'),
    )

    plt.xlabel(xlabel='Fitted values', fontdict=dict(size=12))
    plt.ylabel(
        ylabel='Sqrt of Standardized residuals',
        fontdict=dict(size=12),
    )

    # 쿡의 거리(이상치 탐지)
    ax4 = plt.subplot(2, 2, 4)

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

    plt.tight_layout()


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
    이 함수는 선형 회귀 모델의 잔차를 표준화합니다.

    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을
            지정합니다.

    반환:
        훈련셋의 관측값별 표준화 잔차를 절대값의 내림차순으로 반환합니다.
        인덱스는 원래 관측값의 인덱스를 유지합니다.
    """
    resid = pd.Series(data=model.resid)

    stdres = pd.Series(
        data=stats.zscore(a=resid.to_numpy()),
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


# 잔차의 등분산성 검정 함수(이전 이름)
breushpagan = renamed_alias(breusch_pagan, 'breushpagan')


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
        회귀 모델의 표준화된 회귀계수를 반환합니다. GLM 모델은 입력변수만
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
        y = model.model.endog
        result = model.params * (X.std() / y.std())
    else:
        result = model.params * (X.std() / 1)

    return result


# 회귀 모델의 성능 지표 반환 함수
def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    이 함수는 회귀 모델의 다양한 성능 지표를 계산합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.

    반환:
        회귀 모델의 다양한 성능 지표를 데이터프레임으로 반환합니다.
        실제값과 추정값 중 음수가 포함되면 MSLE와 RMSLE는 결측값으로
        처리합니다.
    """
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


# 회귀 모델의 성능 지표 반환 함수(이전 이름)
regmetrics = renamed_alias(reg_metrics, 'regmetrics')


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


# 분류 모델의 성능 지표 반환 함수
def clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    이 함수는 분류 모델의 다양한 성능 지표를 계산합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.

    반환:
        분류 모델의 다양한 성능 지표를 출력합니다. ipywidgets 패키지가
        설치되어 있으면 혼동행렬과 성능 지표를 가로로 나란히 출력하고,
        설치되어 있지 않으면 세로로 출력합니다.
    """
    y_labels = sorted(pd.Series(data=y_true).unique())
    cfm_labels = y_labels + ['All']
    cfm = pd.crosstab(index=y_true, columns=y_pred, margins=True)
    cfm = cfm.reindex(index=cfm_labels, columns=cfm_labels, fill_value=0)
    cfm.index = [f'True_{i}' for i in cfm_labels]
    cfm.columns = [f'Pred_{i}' for i in cfm_labels]

    report = metrics.classification_report(
        y_true=y_true,
        y_pred=y_pred,
        digits=4,
    )

    widgets = try_import('ipywidgets')
    ipython = try_import('IPython.display')

    # ipywidgets가 없으면 혼동행렬과 성능 지표를 세로로 출력
    if widgets is None or ipython is None:
        print('▶ Confusion Matrix')
        print(cfm.to_string())
        print()
        print('▶ Classification Report')
        print(report)
        return

    display = ipython.display

    left = widgets.Output()
    right = widgets.Output()

    with left:
        print('▶ Confusion Matrix')
        display(cfm)
    with right:
        print('▶ Classification Report')
        print(report)

    left.layout = widgets.Layout(margin='0px 10px 0px 0px')
    right.layout = widgets.Layout(margin='0px 0px 0px 10px')

    box = widgets.HBox(children=[left, right])
    display(box)


# 분류 모델의 성능 지표 반환 함수(이전 이름)
clfmetrics = renamed_alias(clf_metrics, 'clfmetrics')


# 분류 모델의 분류 기준점별 성능 지표 계산(TPR, FPR, MCC)
def clf_cutoffs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: str | int = None,
) -> pd.DataFrame:
    """
    이 함수는 분류 모델에 대한 최적의 분류 기준점을 탐색합니다.

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


# End of Document
