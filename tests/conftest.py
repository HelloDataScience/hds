# 테스트 공통 설정
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


# 테스트마다 열린 그래프를 닫는 픽스처
@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


# 선형 회귀 모델용 데이터 픽스처
@pytest.fixture
def reg_data():
    rng = np.random.default_rng(seed=0)
    n = 200

    # train_test_split 이후처럼 순서가 섞인 인덱스를 사용
    index = rng.permutation(1000)[:n]

    X = pd.DataFrame(
        data={
            'x1': rng.normal(size=n),
            'x2': rng.normal(size=n),
            'x3': rng.normal(size=n),
        },
        index=index,
    )

    y = pd.Series(1 + 2 * X['x1'] - X['x2'] + rng.normal(size=n), name='y')

    return y, X


# 로지스틱 회귀 모델용 데이터 픽스처(양성 범주 1이 다수 범주)
@pytest.fixture
def logit_data():
    rng = np.random.default_rng(seed=1)
    n = 400

    X = pd.DataFrame(
        data={
            'x1': rng.normal(size=n),
            'x2': rng.normal(size=n),
        }
    )

    y = pd.Series(
        data=(X['x1'] + rng.normal(size=n) > -0.8).astype(int),
        name='target',
    )

    return y, X


# 이진 분류 모델의 실제값과 2차원 예측 확률 픽스처
@pytest.fixture
def clf_data(logit_data):
    y, X = logit_data
    proba = LogisticRegression().fit(X=X, y=y).predict_proba(X=X)

    return y, proba
