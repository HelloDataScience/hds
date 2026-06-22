# 관련 라이브러리 호출
import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sma
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oi
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
import ipywidgets as widgets
from IPython.display import display

# 선형 회귀 모델을 적합하는 함수
def ols(y: pd.Series, X: pd.DataFrame) -> statsmodels.api.OLS:
    '''
    이 함수는 선형 회귀 모델을 적합합니다.
    
    매개변수:
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
    
    반환:
        선형 회귀 모델을 반환합니다.
    '''
    if 'const' not in X.columns:
        X.insert(loc = 0, column = 'const', value = 1)
    
    model = sma.OLS(endog = y, exog = X)
    
    return model.fit()


# 선형 회귀 모델을 전진선택법으로 적합하는 함수
def forward_selection(y: pd.Series, X: pd.DataFrame) -> statsmodels.formula.api.ols:
    '''
    이 함수는 다중 선형 회귀 모델을 전진선택법으로 적합합니다.
    
    매개변수:
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
    
    반환:
        전진선택법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    '''
    if 'const' in X.columns:
        X = X.drop(labels = ['const'], axis = 1)
    
    Xvars = list(set(X.columns))
    dat = pd.concat(objs = [X, y], axis = 1)
    formula = f'{y.name} ~ 1'
    curr_aic = smf.ols(formula = formula, data = dat).fit().aic
    
    selected = []
    
    while Xvars:
        Xvar_aic = []
        for Xvar in Xvars:
            formula = f'{y.name} ~ {" + ".join(selected + [Xvar])} + 1'
            aic = smf.ols(formula = formula, data = dat).fit().aic
            aic = np.round(a = aic, decimals = 4)
            Xvar_aic.append((aic, Xvar))
        
        Xvar_aic.sort(reverse = True)
        new_aic, best_Xvar = Xvar_aic.pop()
        
        if curr_aic > new_aic:
            Xvars.remove(best_Xvar)
            selected.append(best_Xvar)
            curr_aic = new_aic
        else:
            break
    
    formula = f'{y.name} ~ {" + ".join(selected)} + 1'
    model = smf.ols(formula = formula, data = dat).fit()
    
    return model


# 선형 회귀 모델을 후진소거법으로 적합하는 함수
def backward_selection(y: pd.Series, X: pd.DataFrame) -> statsmodels.formula.api.ols:
    '''
    이 함수는 선형 회귀 모델을 후진소거법으로 적합합니다.
    
    매개변수:
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
    
    반환:
        후진소거법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    '''    
    if 'const' in X.columns:
        X = X.drop(labels = ['const'], axis = 1)
    
    Xvars = list(set(X.columns))
    dat = pd.concat(objs = [X, y], axis = 1)
    formula = f'{y.name} ~ {" + ".join(list(Xvars))}'
    curr_aic = smf.ols(formula = formula, data = dat).fit().aic
    
    selected = []
    
    while Xvars:
        Xvar_aic = []
        for Xvar in Xvars:
            sub = dat.drop(labels = selected + [Xvar], axis = 1).copy()
            sub_Xvars = set(sub.columns) - set([y.name])
            formula = f'{y.name} ~ {" + ".join(list(sub_Xvars))} + 1'
            aic = smf.ols(formula = formula, data = sub).fit().aic
            aic = np.round(a = aic, decimals = 4)
            Xvar_aic.append((aic, Xvar))
        
        Xvar_aic.sort(reverse = True)
        new_aic, best_Xvar = Xvar_aic.pop()
        
        if curr_aic > new_aic:
            Xvars.remove(best_Xvar)
            selected.append(best_Xvar)
            curr_aic = new_aic
        else:
            break
    
    dat = dat.drop(labels = selected, axis = 1)
    dat_Xvars = set(dat.columns) - set([y.name])
    formula = f'{y.name} ~ {" + ".join(list(dat_Xvars))} + 1'
    model = smf.ols(formula = formula, data = dat).fit()
    
    return model


# 선형 회귀 모델을 단계적방법으로 적합하는 함수
def stepwise_selection(y: pd.Series, X: pd.DataFrame) -> statsmodels.formula.api.ols:
    '''
    이 함수는 선형 회귀 모델을 단계적방법으로 적합합니다.
    
    매개변수:
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
    
    반환:
        단계적방법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    '''
    if 'const' in X.columns:
        X = X.drop(labels = ['const'], axis = 1)
    
    Xvars = list(set(X.columns))
    dat = pd.concat(objs = [X, y], axis = 1)
    formula = f'{y.name} ~ 1'
    curr_aic = smf.ols(formula = formula, data = dat).fit().aic
    
    selected = []
    
    while Xvars:
        Xvar_aic = []
        for Xvar in Xvars:
            formula = f'{y.name} ~ {" + ".join(selected + [Xvar])} + 1'
            aic = smf.ols(formula = formula, data = dat).fit().aic
            aic = np.round(a = aic, decimals = 4)
            Xvar_aic.append((aic, 'add', Xvar))
        
        if selected:
            for Xvar in selected:
                sub = dat[selected + [y.name]].copy()
                sub = sub.drop(labels = [Xvar], axis = 1)
                sub_Xvars = set(sub.columns) - set([y.name])
                formula = f'{y.name} ~ {" + ".join(list(sub_Xvars))} + 1'
                aic = smf.ols(formula = formula, data = sub).fit().aic
                aic = np.round(a = aic, decimals = 4)
                Xvar_aic.append((aic, 'sub', Xvar))
        
        Xvar_aic.sort(reverse = True)
        new_aic, how, best_Xvar = Xvar_aic.pop()
        
        if curr_aic > new_aic and how == 'add':
            Xvars.remove(best_Xvar)
            selected.append(best_Xvar)
            curr_aic = new_aic
        elif curr_aic > new_aic and how == 'sub':
            Xvars.append(best_Xvar)
            selected.remove(best_Xvar)
            curr_aic = new_aic
        elif curr_aic <= new_aic:
            break
    
    formula = f'{y.name} ~ {" + ".join(selected)} + 1'
    model = smf.ols(formula = formula, data = dat).fit()
    
    return model


# 선형 회귀 모델을 변수선택법으로 적합하는 함수
def stepwise(y: pd.Series, X: pd.DataFrame, direction: str = 'both') -> statsmodels.formula.api.ols:
    '''
    이 함수는 세 가지 선형 회귀 모델의 변수선택법을 선택하는 함수입니다.
    
    매개변수:
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
        direction: 변수선택법을 'forward', 'backward' 또는 'both'에서 선택합니다.
                   (기본값: 'both')
    
    반환:
        선택한 방법으로 회귀 모델을 적합하고 AIC 값이 최소인 모델을 반환합니다.
        statsmodels.formula.api.ols 함수를 사용합니다.
    '''
    if direction == 'forward':
        model = forward_selection(y, X)
    elif direction == 'backward':
        model = backward_selection(y, X)
    elif direction == 'both':
        model = stepwise_selection(y, X)
    else:
        model = None
    
    return model


# 선형 회귀 모델의 잔차진단 함수
def regressionDiagnosis(model: statsmodels.api.OLS) -> None:
    '''
    이 함수는 선형 회귀 모델의 잔차가정 만족 여부를 확인하는 다양한 그래프를 그립니다.
    
    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을 지정합니다.
    
    반환:
        네 가지 그래프 외에 반환하는 객체는 없습니다.
    '''
    plt.figure(figsize = (10, 10), dpi = 100)
    
    # 선형성 가정
    # 잔차로 lowess(locally weighted linear regression) 회귀선을 산점도에 추가
    ax1 = plt.subplot(2, 2, 1)
    
    sns.regplot(
        x = model.fittedvalues, 
        y = model.resid, 
        lowess = True, 
        scatter_kws = dict(
            color = '0.8',
            ec = '0.3', 
            s = 15
        ),
        line_kws = dict(
            color = 'red', 
            linewidth = 1
        ), 
        ax = ax1
    )
    
    plt.axhline(
        y = 0, 
        color = '0.5', 
        linestyle = '--', 
        linewidth = 1
    )
    
    plt.title(
        label = 'Residuals vs Fitted', 
        fontdict = dict(size = 14, 
                        fontweight = 'bold')
    )
    
    plt.xlabel(
        xlabel = 'Fitted values', 
        fontdict = dict(size = 12)
    )
    
    plt.ylabel(
        ylabel = 'Residuals', 
        fontdict = dict(size = 12)
    )
    
    # 정규성 가정 확인
    ax2 = plt.subplot(2, 2, 2)
    
    # 표준화 잔차(Standardized residuals)
    stdres = pd.Series(data = stats.zscore(a = model.resid))
    
    # 이론상 분위수(Theoretical Quantiles)
    (x, y), _ = stats.probplot(x = stdres)
    
    # Q-Q plot
    sns.scatterplot(
        x = x, 
        y = y, 
        color = '0.8', 
        ec = '0.3', 
        size = 2, 
        legend = False, 
        ax = ax2
    )
    
    plt.plot(
        [-4, 4], 
        [-4, 4], 
        color = '0.5', 
        linestyle = '--', 
        linewidth = 1
    )
    
    plt.title(
        label = 'Normal Q-Q', 
        fontdict = dict(size = 14, 
                        fontweight = 'bold')
    )
    
    plt.xlabel(
        xlabel = 'Theoretical Quantiles', 
        fontdict = dict(size = 12)
    )
    
    plt.ylabel(
        ylabel = 'Standardized residuals', 
        fontdict = dict(size = 12)
    )
    
    # 등분산성 가정 확인
    ax3 = plt.subplot(2, 2, 3)
    
    sns.regplot(
        x = model.fittedvalues, 
        y = np.sqrt(stdres.abs()), 
        lowess = True,
        scatter_kws = dict(
            color = '0.8', 
            ec = '0.3', 
            s = 15
        ),
        line_kws = dict(
            color = 'red', 
            linewidth = 1
        ), 
        ax = ax3
    )
    
    plt.title(
        label = 'Scale-Location', 
        fontdict = dict(size = 14, 
                        fontweight = 'bold')
    )
    
    plt.xlabel(
        xlabel = 'Fitted values', 
        fontdict = dict(size = 12)
    )
    
    plt.ylabel(
        ylabel = 'Sqrt of Standardized residuals', 
        fontdict = dict(size = 12)
    )

    # 쿡의 거리(이상치 탐지)
    ax4 = plt.subplot(2, 2, 4)
    
    fig = sma.graphics.influence_plot(
        results = model, 
        criterion = 'cooks', 
        size = 24, 
        plot_alpha = 0.2, 
        ax = ax4
    )
    
    for text in ax4.texts:
        text.set_fontsize(8)
        text.set_ha('center')
        text.set_va('center')
    
    plt.tight_layout();


# 쿡의 거리 계산 함수
def cooks_distance(model: statsmodels.api.OLS) -> pd.DataFrame:
    '''
    이 함수는 선형 회귀 모델의 훈련셋으로 관측값별 쿡의 거리를 계산합니다.
    
    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을 지정합니다.
    
    반환:
        훈련셋의 관측값별 쿡의 거리를 반환합니다.
    '''
    cd, _ = oi.OLSInfluence(results = model).cooks_distance
    cd = cd.sort_values(ascending = False)
    
    return cd


# 햇 매트릭스 계산 함수
def hat_matrix(X: pd.DataFrame) -> np.ndarray:
    '''
    이 함수는 입력변수 행렬로 햇 매트릭스(hat matrix)를 계산합니다.
    
    매개변수:
        X: 입력변수 행렬을 pd.DataFrame으로 지정합니다.
    
    반환:
        훈련셋의 햇 매트릭스를 반환합니다.
    '''
    if 'const' not in X.columns:
        X.insert(loc = 0, column = 'const', value = 1)
    
    X = np.array(object = X)
    XtX = np.matmul(X.transpose(), X)
    XtX_inv = np.linalg.inv(XtX)
    result = np.matmul(np.matmul(X, XtX_inv), X.transpose())
    
    return result


# 레버리지(hat value) 계산 함수
def leverage(X: pd.DataFrame) -> pd.DataFrame:
    '''
    이 함수는 입력변수 행렬로 레버리지(hat value)를 계산합니다.
    
    매개변수:
        X: 입력변수 행렬을 pd.DataFrame으로 지정합니다.
    
    반환:
        훈련셋의 관측값별 Leverage를 반환합니다.
    '''
    if 'const' not in X.columns:
        X.insert(loc = 0, column = 'const', value = 1)
    
    n = X.shape[0]
    hatMat = hat_matrix(X = X)
    X['Leverage'] = np.array([hatMat[i][i] for i in range(n)])
    X = X.iloc[:, -1].sort_values(ascending = False)
    
    return X


# 표준화 잔차 계산 함수
def std_Resid(model: statsmodels.api.OLS) -> pd.Series:
    '''
    이 함수는 선형 회귀 모델의 잔차를 표준화합니다.
    
    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을 지정합니다.
    
    반환:
        훈련셋의 관측값별 표준화 잔차를 반환합니다.
    '''
    stdres = pd.Series(data = stats.zscore(a = model.resid))
    locs = stdres.abs().sort_values(ascending = False)
    
    return stdres[locs.index]


# 선형 회귀 모델의 영향점 계산 함수
def augment(model: statsmodels.api.OLS) -> pd.DataFrame:
    '''
    이 함수는 선형 회귀 모델의 영향점을 계산합니다.
    
    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을 지정합니다.
    
    반환:
        선형 회귀 모델의 영향점에 관련한 여러 지표를 데이터프레임으로 반환합니다.
    '''    
    infl = model.get_influence()
    
    df1 = pd.DataFrame(
        data = {model.model.endog_names: infl.endog},
        index = model.fittedvalues.index
    )
    
    df2 = pd.DataFrame(
        data = {
            'fitted': model.fittedvalues,
            'resid': model.resid,
            'hat': infl.hat_matrix_diag,
            'sigma': np.sqrt(infl.sigma2_not_obsi),
            'cooksd': infl.cooks_distance[0],
            'std_resid': infl.resid_studentized
        }
    )
    
    result = pd.concat(objs = [df1, df2], axis = 1)
    
    return result


# 잔차의 등분산성 검정 함수
def breushpagan(model: statsmodels.api.OLS) -> pd.DataFrame:
    '''
    이 함수는 선형 회귀 모델의 잔차 등분산성 검정을 실행합니다.
    
    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을 지정합니다.
    
    반환:
        선형 회귀 모델의 잔차 등분산성 검정 결과를 반환합니다.
    '''
    test = sma.stats.het_breuschpagan(
        resid = model.resid, 
        exog_het = model.model.exog
    )
    
    result = pd.DataFrame(
        data = test, 
        index = ['Statistic', 'P-Value', 'F-Value', 'F P-Value']
    ).T
    
    return result


# 분산팽창지수 반환 함수
def vif(model: statsmodels.api.OLS) -> pd.DataFrame:
    '''
    이 함수는 입력변수 행렬의 분산팽창지수를 계산합니다.
    
    매개변수:
         model: statsmodels.formula.api 모듈 함수로 적합한 회귀 모델을 지정합니다.
    
    반환:
        입력변수 행렬의 열별 분산팽창지수를 반환합니다.
    '''
    func = oi.variance_inflation_factor
    ncol = len(model.model.exog_names)
    vifs = [func(exog = model.model.exog, exog_idx = i) for i in range(1, ncol)]
    result = pd.DataFrame(data = vifs, index = model.model.exog_names[1:]).T
    
    return result


# 회귀계수 반환 함수
def coefs(model: statsmodels.api.OLS) -> pd.Series:
    '''
    이 함수는 회귀 모델의 회귀계수를 확인합니다.
    
    매개변수:
        model: statsmodels.formula.api.ols 함수로 적합한 선형 회귀 모델을 지정합니다.
    
    반환:
        회귀 모델의 회귀계수를 반환합니다.
    '''
    if model.coef_.ndim == 1:
        coefs = pd.Series(
            data = model.coef_, 
            index = model.feature_names_in_
        )
    elif model.coef_.ndim == 2:
        coefs = pd.Series(
            data = model.coef_[0], 
            index = model.feature_names_in_
        )
    else:
        coefs = pd.Series()
    
    return coefs


# 표준화된 회귀계수 반환 함수
def std_coefs(model: statsmodels.api.OLS) -> pd.Series:
    '''
    이 함수는 회귀 모델의 표준화된 회귀계수를 계산합니다.
    
    매개변수:
        model: statsmodels.formula.api 모듈 함수로 적합한 회귀 모델을 지정합니다.
    
    반환:
        회귀 모델의 표준화된 회귀계수를 반환합니다.
    '''
    model_type = str(type(model.model))
    
    X = pd.DataFrame(
        data = model.model.exog, 
        columns = model.model.exog_names
    )
    
    if 'OLS' in model_type:
        y = model.model.endog
        result = model.params * (X.std() / y.std())
    elif 'GLM' in model_type:
        y = 1
        result = model.params * (X.std() / 1)
    
    return result


# 회귀 모델의 성능 지표 반환 함수
def regmetrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    '''
    이 함수는 회귀 모델의 다양한 성능 지표를 계산합니다.
    
    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
    
    반환:
        회귀 모델의 다양한 성능 지표를 데이터프레임으로 반환합니다.
        실제값과 추정값 중 음수가 포함되면 MSLE와 RMSLE는 결측값으로 처리합니다.
    '''
    R_2 = metrics.r2_score(y_true = y_true, y_pred = y_pred)
    MSE = metrics.mean_squared_error(y_true = y_true, y_pred = y_pred)
    RMSE = metrics.root_mean_squared_error(y_true = y_true, y_pred = y_pred)
    MAE = metrics.mean_absolute_error(y_true = y_true, y_pred = y_pred)
    
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
        MAPE = metrics.mean_absolute_percentage_error(y_true = y_true, y_pred = y_pred)
    
    result = pd.DataFrame(
        data = {
            'metric': ['R^2', 'MSE', 'RMSE', 'MSLE', 'RMSLE', 'MAE', 'MAPE'],
            'score': [R_2, MSE, RMSE, MSLE, RMSLE, MAE, MAPE], 
            'description': ['R Squared', 'Mean Squared Error', 'Root Mean Squared Error', 
                            'Mean Squared Log Error', 'Root Mean Squared Log Error', 
                            'Mean Absolute Error', 'Mean Absolute Percentage Error']
        }
    )
    
    return result


# 로지스틱 회귀 모델을 적합하는 함수
def glm(y: pd.Series, X: pd.DataFrame) -> statsmodels.api.GLM:
    '''
    이 함수는 로지스틱 회귀 모델을 적합합니다.
    
    매개변수:
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        X: 입력변수 행렬을 pd.DataFrame 또는 2차원 np.ndarray로 지정합니다.
        family: 목표변수의 확률분포를 지정합니다.
    
    반환:
        로지스틱 회귀 모델을 반환합니다.
    '''
    if 'const' not in X.columns:
        X.insert(loc = 0, column = 'const', value = 1)
    
    model = sma.GLM(endog = y, exog = X, family = sma.families.Binomial())
    
    return model.fit()


# 분류 모델의 성능 지표 반환 함수
def clfmetrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    '''
    이 함수는 분류 모델의 다양한 성능 지표를 계산합니다.
    
    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
    
    반환:
        분류 모델의 다양한 성능 지표를 출력합니다.
    '''
    y_labels = sorted(pd.Series(data = y_true).unique())
    cfm_labels = y_labels + ['All']
    cfm = pd.crosstab(index = y_true, columns = y_pred, margins = True)
    cfm = cfm.reindex(index = cfm_labels, columns = cfm_labels, fill_value = 0)
    cfm.index = [f'True_{i}' for i in cfm_labels]
    cfm.columns = [f'Pred_{i}' for i in cfm_labels]
    
    left = widgets.Output()
    right = widgets.Output()

    with left:
        print('▶ Confusion Matrix')
        display(cfm)
    with right:
        print('▶ Classification Report')
        print(
            metrics.classification_report(
                y_true = y_true, 
                y_pred = y_pred, 
                digits = 4
            )
        )
    
    left.layout = widgets.Layout(margin = '0px 10px 0px 0px')
    right.layout = widgets.Layout(margin = '0px 0px 0px 10px')
    
    box = widgets.HBox(children = [left, right])
    display(box)


# 분류 모델의 분류 기준점별 성능 지표 계산(TPR, FPR, Matthew's Correlation coefficient)
def clfCutoffs(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    '''
    이 함수는 분류 모델에 대한 최적의 분류 기준점을 탐색합니다.
    
    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
    
    반환:
        분류 모델의 분류 기준점별로 TPR, FPR, MCC 등을 반환합니다.
    '''
    cutoffs = np.linspace(0, 1, 101)
    sens = []
    spec = []
    prec = []
    mccs = []
    
    for cutoff in cutoffs:
        pred = np.where(y_prob >= cutoff, 1, 0)
        clfr = metrics.classification_report(
            y_true = y_true, 
            y_pred = pred, 
            output_dict = True, 
            zero_division = True
        )
        sens.append(clfr['1']['recall'])
        spec.append(clfr['0']['recall'])
        prec.append(clfr['1']['precision'])
        
        mcc = metrics.matthews_corrcoef(
            y_true = y_true, 
            y_pred = pred
        )
        mccs.append(mcc)
        
    result = pd.DataFrame(
        data = {
            'Cutoff': cutoffs, 
            'Sensitivity': sens, 
            'Specificity': spec, 
            'Precision': prec, 
            'MCC': mccs
        }
    )
    
    # The Optimal Point is the sum of Sensitivity and Specificity.
    result['Optimal'] = result['Sensitivity'] + result['Specificity']
    
    # TPR and FPR for ROC Curve.
    result['TPR'] = result['Sensitivity']
    result['FPR'] = 1 - result['Specificity']
    
    # Set Column name.
    cols = ['Cutoff', 'Sensitivity', 'Specificity', 'Optimal', \
            'Precision', 'TPR', 'FPR', 'MCC']

    # Select columns
    result = result[cols]
    
    return result


# 최적의 분류 기준점 시각화 함수
def EpiROC(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    '''
    이 함수는 분류 모델에 대한 최적의 분류 기준점을 ROC 곡선에 추가합니다.
    
    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_pred: 목표변수의 추정값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
    
    반환:
         ROC 곡선 그래프 외에 반환하는 객체는 없습니다.
    '''    
    obj = clfCutoffs(y_true, y_prob)
    
    # Draw ROC curve
    sns.lineplot(
        data = obj, 
        x = 'FPR', 
        y = 'TPR', 
        color = 'black'
    )

    # Add title
    plt.title(label = '최적의 분류 기준점 탐색', 
              fontdict = {'fontweight': 'bold'})
    
    # Draw diagonal line
    plt.plot(
        [0, 1], 
        [0, 1], 
        color = '0.5', 
        linestyle = '--', 
        linewidth = 0.5
    )
    
    # Add the Optimal Point
    opt = obj.iloc[[obj['Optimal'].argmax()]]
    
    sns.scatterplot(
        data = opt, 
        x = 'FPR', 
        y = 'TPR', 
        color = 'red'
    )
    
    # Add tangent line
    optX = opt['FPR'].iloc[0]
    optY = opt['TPR'].iloc[0]
    
    b = optY - optX
    
    plt.plot(
        [0, 1-b], 
        [b, 1], 
        color = 'red', 
        linestyle = '-.', 
        linewidth = 0.5
    )
    
    # Add text
    plt.text(
        x = opt['FPR'].values[0] - 0.01, 
        y = opt['TPR'].values[0] + 0.01, 
        s = f"Cutoff = {opt['Cutoff'].round(2).values[0]}", 
        ha = 'right', 
        va = 'bottom'
    );


## End of Document