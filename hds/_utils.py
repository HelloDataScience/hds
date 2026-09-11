# 관련 라이브러리 호출
import importlib
from types import ModuleType

import numpy as np
import pandas as pd


# 선택 의존성 패키지를 호출하는 함수
def import_optional(
    module: str,
    extra: str,
    package: str = None,
    hint: str = None,
) -> ModuleType:
    """
    이 함수는 선택 의존성 패키지를 호출합니다. 해당 패키지가 설치되어 있지
    않으면 설치 방법을 안내하는 예외를 발생시킵니다.

    매개변수:
        module: 호출할 모듈명을 문자열로 지정합니다.
        extra: 해당 모듈이 포함된 추가 설치 옵션명을 문자열로 지정합니다.
        package: PyPI에 등록된 패키지명이 모듈명과 다르면 문자열로
            지정합니다.(기본값: None)
        hint: 안내 문구에 덧붙일 설명을 문자열로 지정합니다.(기본값: None)

    반환값:
        호출한 모듈을 반환합니다.
    """
    try:
        return importlib.import_module(name=module)
    except ImportError as error:
        name = package if package is not None else module.split('.')[0]
        message = (
            f"'{name}' 패키지가 설치되어 있지 않습니다. "
            f"pip install 'hds[{extra}]' 명령으로 설치하세요."
        )
        if hint is not None:
            message = f'{message} {hint}'
        raise ImportError(message) from error


# 선택 의존성 패키지를 호출하고 없으면 None을 반환하는 함수
def try_import(module: str) -> ModuleType:
    """
    이 함수는 선택 의존성 패키지를 호출하고, 설치되어 있지 않으면 예외 대신
    None을 반환합니다.

    매개변수:
        module: 호출할 모듈명을 문자열로 지정합니다.

    반환값:
        호출한 모듈을 반환하고, 설치되어 있지 않으면 None을 반환합니다.
    """
    try:
        return importlib.import_module(name=module)
    except ImportError:
        return None


# 입력변수 행렬을 원본과 분리된 데이터프레임으로 변환하는 함수
def as_frame(X: pd.DataFrame, index: pd.Index = None) -> pd.DataFrame:
    """
    이 함수는 입력변수 행렬을 원본과 분리된 데이터프레임으로 변환합니다.
    np.ndarray는 statsmodels와 같이 열 이름을 x1, x2, ... 순서로 지정합니다.

    매개변수:
        X: 입력변수 행렬을 pd.DataFrame, pd.Series 또는 np.ndarray로
            지정합니다.
        index: X가 np.ndarray일 때 행 인덱스로 사용할 인덱스를 지정합니다.
            행 개수가 다르면 무시합니다.(기본값: None)

    반환값:
        원본과 분리된 데이터프레임을 반환합니다.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    if isinstance(X, pd.Series):
        name = 'x1' if X.name is None else X.name
        return X.to_frame(name=name).copy()

    values = np.asarray(X)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if values.ndim != 2:
        raise ValueError('입력변수 행렬은 2차원 배열로 지정해야 합니다.')

    if index is not None and len(index) != values.shape[0]:
        index = None

    columns = [f'x{i}' for i in range(1, values.shape[1] + 1)]

    return pd.DataFrame(data=values, index=index, columns=columns)


# 이진 분류의 양성 범주를 결정하는 함수
def resolve_pos_label(y_true: pd.Series, pos_label: object = None) -> object:
    """
    이 함수는 분류 모델의 성능을 계산할 양성 범주를 결정합니다. pos_label을
    생략하면 목표변수의 범주가 0과 1이면 1, False와 True이면 True를 양성
    범주로 사용합니다. 그 밖의 범주는 양성 범주를 추측하지 않고 예외를
    발생시킵니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        pos_label: 양성 범주를 지정합니다.(기본값: None)

    반환값:
        양성 범주를 반환합니다.
    """
    y_true = pd.Series(data=y_true)
    classes = y_true.dropna().unique().tolist()

    if pos_label is not None:
        if pos_label not in classes:
            raise ValueError(
                f'pos_label({pos_label!r})이 y_true의 범주 {classes}에 '
                '없습니다.'
            )
        return pos_label

    if classes and set(classes) <= {0, 1}:
        return True if y_true.dtype == bool else 1

    raise ValueError(
        f'y_true의 범주가 {classes}이므로 양성 범주를 판단할 수 없습니다. '
        'pos_label 매개변수로 양성 범주를 지정하세요.'
    )


# 양성 범주의 예측 확률을 1차원 배열로 반환하는 함수
def pos_proba(
    y_true: pd.Series,
    y_prob: np.ndarray,
    pos_label: object,
) -> np.ndarray:
    """
    이 함수는 예측 확률에서 양성 범주의 확률을 1차원 배열로 추출합니다.
    2차원 예측 확률은 scikit-learn의 predict_proba() 함수처럼 열이 범주의
    오름차순으로 정렬되어 있다고 가정합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        y_prob: 목표변수의 예측 확률을 1차원 또는 2차원 np.ndarray로
            지정합니다.
        pos_label: 양성 범주를 지정합니다.

    반환값:
        양성 범주의 예측 확률을 1차원 np.ndarray로 반환합니다.
    """
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        return y_prob

    if y_prob.ndim != 2:
        raise ValueError('y_prob은 1차원 또는 2차원 배열로 지정해야 합니다.')

    classes = sorted(pd.Series(data=y_true).dropna().unique().tolist())

    if y_prob.shape[1] != len(classes):
        raise ValueError(
            f'y_prob의 열 개수({y_prob.shape[1]})와 y_true의 범주 개수'
            f'({len(classes)})가 다릅니다. 양성 범주의 확률만 1차원 배열로 '
            '지정하세요.'
        )

    return y_prob[:, classes.index(pos_label)]


# End of Document
