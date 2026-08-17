# 관련 라이브러리 호출
import glob
import inspect
import json
import os
import platform
import re
import shutil
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, enet_path
from sklearn.preprocessing import StandardScaler
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_graphviz,
)

from hds._utils import import_optional, try_import


# 그래프를 그릴 Axes를 준비하는 함수
def _prepare_ax(ax: plt.Axes) -> tuple:
    """
    이 함수는 그래프를 그릴 Axes 객체를 준비합니다. ax를 생략하면 현재 Axes를
    사용하고 그래프를 화면에 출력하도록 설정합니다.

    매개변수:
        ax: matplotlib Axes 객체를 지정합니다.

    반환값:
        Axes 객체와 화면 출력 여부를 튜플로 반환합니다.
    """
    if ax is None:
        return plt.gca(), True
    return ax, False


# 호출부에서 인수로 지정한 변수명을 반환하는 함수
def _arg_name(arg: str) -> str:
    """
    이 함수는 varname 패키지가 설치되어 있으면 호출부에서 인수로 지정한
    변수명을 반환합니다. varname 패키지가 없거나 변수명을 확인할 수 없으면
    None을 반환합니다.

    매개변수:
        arg: 변수명을 확인할 매개변수명을 문자열로 지정합니다.

    반환값:
        호출부에서 인수로 지정한 변수명을 문자열로 반환합니다.
    """
    varname = try_import('varname')
    if varname is None:
        return None
    try:
        return varname.argname(arg, frame=2)
    except Exception:
        return None


# 구글 폰트 파일 목록을 반환하는 함수
def search_google_font_file(font_name: str) -> list:
    """
    이 함수는 구글 폰트(https://fonts.google.com)에 등록된 폰트명을 지정하면
    해당 폰트명의 ttf 파일 목록을 반환합니다.

    매개변수:
        font_name: 구글 폰트명을 문자열로 지정합니다.

    반환값:
        구글 폰트 ttf 파일명을 리스트로 반환합니다.
    """
    # 선택 의존성 패키지 호출
    requests = import_optional('requests', extra='font')
    bs4 = import_optional('bs4', extra='font', package='beautifulsoup4')

    # 구글 폰트명에서 공백 제거
    font_name_no_space = font_name.replace(' ', '')

    # 구글 폰트명 URL 생성
    url = (
        'https://github.com/google/fonts/tree/main/ofl/'
        f'{font_name_no_space.lower()}'
    )

    # 구글 폰트 파일 목록 내려받기
    response = requests.get(url)
    if response.status_code != 200:
        raise FileNotFoundError(f'Font not found with {font_name}')

    soup = bs4.BeautifulSoup(markup=response.text, features='html.parser')
    selector = (
        'script[type="application/json"]'
        '[data-target="react-app.embeddedData"]'
    )
    items = soup.select(selector)
    payload = json.loads(s=items[0].text)
    files = payload['payload']['tree']['items']
    return [item['name'] for item in files if '.ttf' in item['name']]


# 구글 폰트 파일을 다운로드 폴더에 내려받는 함수
def download_google_font_file(font_file: str) -> str:
    """
    이 함수는 구글 폰트 ttf 파일명을 지정하면 다운로드 폴더에 내려받습니다.

    매개변수:
        font_file: 구글 폰트 ttf 파일명을 문자열로 지정합니다.

    반환값:
        다운로드 폴더에 내려받은 구글 폰트 ttf 파일명을 문자열로 반환합니다.
    """
    # 선택 의존성 패키지 호출
    requests = import_optional('requests', extra='font')

    # 다운로드 폴더 경로 지정
    download_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    os.makedirs(name=download_path, exist_ok=True)
    font_path = os.path.join(download_path, font_file)

    # 구글 폰트명 생성
    font_name = re.split(
        pattern=r'(-)|(\[)|(\.ttf)',
        string=font_file,
    )[0].lower()

    # 구글 폰트 파일 다운로드 URL 생성
    domain = (
        'https://raw.githubusercontent.com/google/fonts/'
        'refs/heads/main/ofl/'
    )
    url = os.path.join(domain, font_name, font_file)

    # 구글 폰트 파일 내려받기
    response = requests.get(url)
    if response.status_code != 200:
        raise FileNotFoundError(f'Font not found at {url}')

    with open(file=font_path, mode='wb') as file:
        file.write(response.content)
    print(f'Downloaded to {font_path}')
    return font_path


# 구글 폰트를 설치하고 다운로드 폴더에서 삭제하는 함수
def install_google_font_path(font_path: str) -> None:
    """
    이 함수는 다운로드 폴더에 내려받은 구글 폰트 ttf 파일명을 운영체제에 맞게
    설치하고 다운로드 폴더에 있는 ttf 파일명을 삭제합니다.

    매개변수:
        font_path: 다운로드 폴더에 내려받은 구글 폰트 ttf 파일명을 문자열로
            지정합니다.

    반환값:
        없습니다.
    """
    # 운영체제별 구글 폰트 설치 경로 지정
    system = platform.system()
    if system == 'Windows':
        fonts_dir = os.path.join(os.getenv(key='WINDIR'), 'Fonts')
        shutil.copy(src=font_path, dst=fonts_dir)
    elif system == 'Darwin':
        fonts_dir = os.path.expanduser('~/Library/Fonts')
        shutil.copy(src=font_path, dst=fonts_dir)
    elif system == 'Linux':
        fonts_dir = os.path.expanduser('~/.fonts')
        os.makedirs(name=fonts_dir, exist_ok=True)
        shutil.copy(src=font_path, dst=fonts_dir)
        subprocess.run(['fc-cache', '-f', '-v'])
    else:
        raise OSError('Unsupported operating system')

    # 실행 완료 문구 출력
    print(f'Installed font at {fonts_dir}')

    # 구글 폰트 파일 삭제
    os.remove(font_path)


# 구글 폰트를 설치하고 matplotlib 임시 폴더의 json 파일을 삭제하는 함수
def add_google_font(font_name: str) -> None:
    """
    이 함수는 구글 폰트명을 지정하면 해당 폰트의 ttf 파일명을 다운로드 폴더에
    내려받은 다음 운영체제에 맞게 설치하고 다운로드 폴더에서 삭제합니다.

    매개변수:
        font_name: 구글 폰트명을 문자열로 지정합니다.

    반환값:
        없습니다.
    """
    # 구글 폰트 파일 목록 생성
    font_files = search_google_font_file(font_name)

    # 반복문 실행
    for font_file in font_files:
        try:
            # 구글 폰트 파일을 다운로드 폴더에 내려받기
            font_path = download_google_font_file(font_file)

            # 구글 폰트를 설치하고 다운로드 폴더에서 삭제
            install_google_font_path(font_path)
        except Exception as error:
            print(f'Error: {error}')

    # matplotlib 임시 폴더에 있는 json 파일 삭제
    cache_dir = matplotlib.get_cachedir()
    font_list = glob.glob(f'{cache_dir}/fontlist-*.json')[0]
    os.remove(path=font_list)


# 범례가 있을 때만 제거하는 함수
def remove_legend(ax: plt.Axes = None) -> None:
    """
    이 함수는 Axes에 범례가 있을 때만 제거합니다.

    매개변수:
        ax: matplotlib Axes 객체를 지정합니다. 생략하면 현재 Axes를
            사용합니다.(기본값: None)

    반환값:
        없습니다.
    """
    if ax is None:
        ax = plt.gca()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


# 집단별 상자 수염 그림을 그리는 함수
def box_group(
    data: pd.DataFrame,
    x: str,
    y: str,
    palette: list = None,
    legend: bool = False,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 범주형 변수(x축)에 따라 연속형 변수(y축)의 상자 수염 그림을
    그립니다. 상자에 빨간 점은 해당 범주의 평균이며, 가로 직선은 전체
    평균입니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        y: 연속형 변수명을 문자열로 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.
        legend: 범례 추가 여부를 True 또는 False로 지정합니다.(기본값: False)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    avg = data.groupby(by=x)[y].mean()

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=x,
        order=avg.index,
        palette=palette,
        flierprops={
            'marker': 'o',
            'markersize': 3,
            'markerfacecolor': 'pink',
            'markeredgecolor': 'red',
            'markeredgewidth': 0.2,
        },
        linecolor='0.5',
        linewidth=0.5,
        ax=ax,
    )

    ax.axhline(
        y=data[y].mean(),
        color='red',
        linewidth=0.5,
        linestyle='--',
    )

    for i, value in enumerate(avg):
        ax.text(
            x=i,
            y=value,
            s=f'{value:,.2f}',
            ha='center',
            va='center',
            fontsize=6,
            fontweight='bold',
        )

    if legend is True:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=x)
    else:
        remove_legend(ax)

    ax.set_title(
        label=f'{x} 범주별 {y}의 평균 비교',
        fontdict={'fontweight': 'bold'},
    )

    if show:
        plt.show()

    return ax


# 두 연속형 변수로 산점도를 그리는 함수
def scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str = '0.3',
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 두 연속형 변수의 산점도를 그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 원인이 되는 연속형 변수명을 문자열로 지정합니다.
        y: 결과가 되는 연속형 변수명을 문자열로 지정합니다.
        color: 점의 채우기 색을 문자열로 지정합니다.(기본값: '0.3')
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    sns.scatterplot(data=data, x=x, y=y, color=color, ax=ax)

    ax.set_title(
        label=f'{x}와(과) {y}의 관계',
        fontdict={'fontweight': 'bold'},
    )

    if show:
        plt.show()

    return ax


# 두 연속형 변수로 산점도와 회귀직선을 그리는 함수
def regline(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str = '0.3',
    size: int = 15,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 두 연속형 변수의 산점도에 회귀직선을 그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 원인이 되는 연속형 변수명을 문자열로 지정합니다.
        y: 결과가 되는 연속형 변수명을 문자열로 지정합니다.
        color: 점의 채우기 색을 문자열로 지정합니다.(기본값: '0.3')
        size: 점의 크기를 정수로 지정합니다.(기본값: 15)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    sns.regplot(
        data=data,
        x=x,
        y=y,
        ci=None,
        scatter_kws={
            'facecolor': color,
            'edgecolor': '1',
            's': size,
            'alpha': 0.2,
        },
        line_kws={
            'color': 'red',
            'linewidth': 1.5,
        },
        ax=ax,
    )

    ax.set_title(
        label=f'{x}와(과) {y}의 관계',
        fontdict={'fontweight': 'bold'},
    )

    if show:
        plt.show()

    return ax


# 범주형 변수의 도수로 막대 그래프를 그리는 함수
def bar_freq(
    data: pd.DataFrame,
    x: str,
    color: str = None,
    palette: list = None,
    legend: bool = False,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 범주형 변수의 도수를 내림차순 정렬한 막대 그래프를 그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        color: 점의 채우기 색을 문자열로 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.
        legend: 범례 추가 여부를 True 또는 False로 지정합니다.(기본값: False)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    counts = data[x].value_counts().sort_index()
    max_count = counts.values.max()
    offset = np.ceil(max_count * 0.01)

    sns.countplot(
        data=data,
        x=x,
        hue=x,
        order=counts.index,
        color=color,
        palette=palette,
        ax=ax,
    )

    for i, value in enumerate(counts):
        ax.text(
            x=i,
            y=value + offset,
            s=value,
            ha='center',
            va='bottom',
            c='black',
            fontsize=8,
            fontweight='bold',
        )

    if legend is True:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=x)
    else:
        remove_legend(ax)

    ax.set_ylim(0, max_count * 1.2)
    ax.set_title(
        label='목표변수의 범주별 도수 비교',
        fontdict={'fontweight': 'bold'},
    )

    if show:
        plt.show()

    return ax


# 범주형 변수를 소그룹으로 나누고 도수로 펼친 막대 그래프를 그리는 함수
def bar_dodge_freq(
    data: pd.DataFrame,
    x: str,
    g: str,
    palette: list = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 범주형 변수를 소그룹으로 나누고 도수로 펼친 막대 그래프를
    그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        g: x를 소그룹으로 나눌 범주형 변수명을 문자열로 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    counts = data.groupby(by=[x, g]).count().iloc[:, 0]
    max_count = counts.values.max()
    offset = np.ceil(max_count * 0.01)

    sns.countplot(
        data=data,
        x=x,
        hue=g,
        order=counts.index.levels[0],
        hue_order=counts.index.levels[1],
        palette=palette,
        ax=ax,
    )

    for i, value in enumerate(counts):
        if i % 2 == 0:
            pos = i / 2 - 0.2
        else:
            pos = (i - 1) / 2 + 0.2
        ax.text(
            x=pos,
            y=value + offset,
            s=value,
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
        )

    ax.set_ylim(0, max_count * 1.2)
    ax.set_title(
        label=f'{x}의 범주별 {g}의 도수 비교',
        fontdict={'fontweight': 'bold'},
    )
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    if show:
        plt.show()

    return ax


# 범주형 변수를 소그룹으로 나누고 도수로 쌓은 막대 그래프를 그리는 함수
def bar_stack_freq(
    data: pd.DataFrame,
    x: str,
    g: str,
    kind: str = 'bar',
    palette: list = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 범주형 변수를 소그룹으로 나누고 도수로 쌓은 막대 그래프를
    그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        g: x를 소그룹으로 나눌 범주형 변수명을 문자열로 지정합니다.
        kind: 막대 그래프의 종류를 'bar' 또는 'barh'로 지정합니다.(기본값: 'bar')
        palette: 팔레트를 리스트로 지정합니다.
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    n_groups = data[g].unique().size

    pivot = pd.pivot_table(
        data=data,
        index=x,
        columns=g,
        aggfunc='count',
    )

    pivot = pivot.iloc[:, 0:n_groups].sort_index()
    pivot.columns = pivot.columns.droplevel(level=0)
    pivot.columns.name = None
    pivot = pivot.reset_index()
    cols = pivot.columns[1:]
    cumsum = pivot[cols].cumsum(axis=1)

    if isinstance(palette, list):
        palette = sns.set_palette(sns.color_palette(palette))

    pivot.plot(
        x=x,
        kind=kind,
        stacked=True,
        rot=0,
        legend='reverse',
        colormap=palette,
        ax=ax,
    )

    if kind == 'bar':
        for col in cols:
            for i, (total, value) in enumerate(zip(cumsum[col], pivot[col])):
                ax.text(
                    x=i,
                    y=total - value / 2,
                    s=value,
                    ha='center',
                    va='center',
                    c='black',
                    fontsize=8,
                    fontweight='bold',
                )
    elif kind == 'barh':
        for col in cols:
            for i, (total, value) in enumerate(zip(cumsum[col], pivot[col])):
                ax.text(
                    x=total - value / 2,
                    y=i,
                    s=value,
                    ha='center',
                    va='center',
                    c='black',
                    fontsize=8,
                    fontweight='bold',
                )

    ax.set_title(
        label=f'{x}의 범주별 {g}의 도수 비교',
        fontdict={'fontweight': 'bold'},
    )
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    if show:
        plt.show()

    return ax


# 범주형 변수를 소그룹으로 나누고 상대도수로 쌓은 막대 그래프를 그리는 함수
def bar_stack_prop(
    data: pd.DataFrame,
    x: str,
    g: str,
    kind: str = 'bar',
    palette: list = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 범주형 변수를 소그룹으로 나누고 상대도수로 쌓은 막대 그래프를
    그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        g: x를 소그룹으로 나눌 범주형 변수명을 문자열로 지정합니다.
        kind: 막대 그래프의 종류를 'bar' 또는 'barh'로 지정합니다.(기본값: 'bar')
        palette: 팔레트를 리스트로 지정합니다.(기본값: None)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    n_groups = data[g].unique().size

    pivot = pd.pivot_table(
        data=data,
        index=x,
        columns=g,
        aggfunc='count',
    )

    pivot = pivot.iloc[:, 0:n_groups].sort_index()
    pivot.columns = pivot.columns.droplevel(level=0)
    pivot.columns.name = None
    pivot = pivot.reset_index()
    cols = pivot.columns[1:]
    row_sum = pivot[cols].apply(func=sum, axis=1)
    pivot[cols] = pivot[cols].div(row_sum, 0) * 100
    cumsum = pivot[cols].cumsum(axis=1)

    if isinstance(palette, list):
        palette = sns.set_palette(sns.color_palette(palette))

    pivot.plot(
        x=x,
        kind=kind,
        stacked=True,
        rot=0,
        legend='reverse',
        colormap=palette,
        mark_right=True,
        ax=ax,
    )

    if kind == 'bar':
        for col in cols:
            for i, (total, value) in enumerate(zip(cumsum[col], pivot[col])):
                label = f'{np.round(value, 1)}%'
                ax.text(
                    x=i,
                    y=total - value / 2,
                    s=label,
                    ha='center',
                    va='center',
                    c='black',
                    fontsize=8,
                    fontweight='bold',
                )
    elif kind == 'barh':
        for col in cols:
            for i, (total, value) in enumerate(zip(cumsum[col], pivot[col])):
                label = f'{np.round(value, 1)}%'
                ax.text(
                    x=total - value / 2,
                    y=i,
                    s=label,
                    ha='center',
                    va='center',
                    c='black',
                    fontsize=8,
                    fontweight='bold',
                )

    ax.set_title(
        label=f'{x}의 범주별 {g}의 상대도수 비교',
        fontdict={'fontweight': 'bold'},
    )
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    if show:
        plt.show()

    return ax


# 연속형 변수 간 상관관계 히트맵 시각화
def corr_heatmap(
    data: pd.DataFrame,
    palette: str = 'RdYlBu',
    fontsize: int = 8,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 연속형 변수 간 상관관계를 히트맵으로 그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.(기본값: 'RdYlBu')
        fontsize: 주석(상관계수)의 글자 크기를 지정합니다.(기본값: 8)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    corr = data.corr(numeric_only=True)
    mask = np.triu(m=np.ones_like(a=corr, dtype=bool), k=1)

    sns.heatmap(
        data=corr,
        cmap=palette,
        annot=True,
        fmt='.2f',
        annot_kws={'fontweight': 'bold', 'fontsize': fontsize},
        linewidth=1,
        mask=mask,
        ax=ax,
    )

    ax.set_title(
        label='변수 간 상관계수 행렬',
        fontdict={'fontweight': 'bold'},
    )

    if show:
        plt.show()

    return ax


# 등고선을 추가한 이차원 커널 밀도 곡선 시각화
def kde2d(
    data: pd.DataFrame,
    x: str,
    y: str,
    frac: float = 0.2,
    seed: int = 0,
    scatter: bool = False,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 두 연속형 변수의 이차원 커널 밀도 곡선을 등고선으로 그립니다.

    매개변수:
        data: 데이터프레임을 지정합니다.
        x: x축에 놓을 변수명을 문자열로 지정합니다.
        y: y축에 놓을 변수명을 문자열로 지정합니다.
        frac: 산점도를 그릴 샘플 비율을 0~1의 실수로 지정합니다.(기본값: 0.2)
        seed: 시드 초기값을 정수로 지정합니다.(기본값: 0)
        scatter: 산점도 추가 여부를 True 또는 False로 지정합니다.(기본값: False)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    levels = np.arange(start=0.05, stop=1.05, step=0.05)
    sns.kdeplot(
        data=data,
        x=x,
        y=y,
        color='0.9',
        fill=True,
        levels=levels,
        ax=ax,
    )
    if scatter:
        data_sample = data.sample(frac=frac, random_state=seed)
        sns.scatterplot(
            data=data_sample,
            x=x,
            y=y,
            c='0',
            s=10,
            alpha=0.2,
            ax=ax,
        )

    ax.set_title(
        label=f'{x}와 {y}의 관계',
        fontdict={'fontweight': 'bold'},
    )

    if show:
        plt.show()

    return ax


# 의사결정나무 모델 시각화
def tree(
    model,
    file_name: str = None,
    class_name: str = None,
) -> None:
    """
    이 함수는 의사결정나무 모델을 시각화하여 png 파일로 저장합니다.

    매개변수:
        model: 사이킷런으로 적합한 의사결정나무 모델을 지정합니다.
        file_name: 입력변수명을 문자열로 지정합니다.(기본값: None)
        class_name: 분류 모델은 목표변수의 범주를 문자열로 지정합니다.(기본값: None)

    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    """
    # 선택 의존성 패키지 호출
    graphviz = import_optional(
        'graphviz',
        extra='tree',
        hint=(
            '파이썬 패키지와 별개로 운영체제에 Graphviz 실행 파일도 '
            '설치되어 있어야 합니다.(https://graphviz.org/download/)'
        ),
    )

    if file_name is None:
        global_objs = inspect.currentframe().f_back.f_globals.items()
        result = [name for name, value in global_objs if value is model]
        file_name = result[0]

    if isinstance(model, DecisionTreeRegressor):
        export_graphviz(
            decision_tree=model,
            out_file=f'{file_name}.dot',
            feature_names=model.feature_names_in_,
            filled=True,
            leaves_parallel=False,
            impurity=True,
        )
    elif isinstance(model, DecisionTreeClassifier):
        if class_name is None:
            class_name = model.classes_.astype(str)
        export_graphviz(
            decision_tree=model,
            out_file=f'{file_name}.dot',
            class_names=class_name,
            feature_names=model.feature_names_in_,
            filled=True,
            leaves_parallel=False,
            impurity=True,
        )

    with open(file=f'{file_name}.dot', mode='rt') as file:
        graph = file.read()
        graph = graphviz.Source(source=graph, format='png')
        graph.render(filename=file_name)

    os.remove(f'{file_name}')
    os.remove(f'{file_name}.dot')


# 입력변수별 중요도 시각화
def feature_importance(
    model,
    palette: str = 'Spectral',
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 입력변수별 중요도를 막대 그래프로 시각화합니다.

    매개변수:
        model: 사이킷런으로 적합한 분류 모델을 지정합니다.
        palette: 팔레트를 문자열로 지정합니다.(기본값: Spectral)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    if 'LGBM' in str(type(model)):
        names = model.feature_name_
    else:
        names = model.feature_names_in_

    importance_df = (
        pd.DataFrame(
            data=model.feature_importances_,
            index=names,
            columns=['importance'],
        )
        .sort_values(by='importance', ascending=False)
        .reset_index()
    )

    sns.barplot(
        data=importance_df,
        x='importance',
        y='index',
        hue='index',
        palette=palette,
        ax=ax,
    )

    for i, row in importance_df.iterrows():
        ax.text(
            x=row['importance'] + 0.01,
            y=i,
            s=f"{row['importance']:.3f}",
            ha='left',
            va='center',
            fontsize=8,
            fontweight='bold',
        )

    ax.set_xlim(0, importance_df['importance'].max() * 1.2)
    ax.set_title(
        label='Feature Importances',
        fontdict={'fontweight': 'bold'},
    )
    ax.set_xlabel(xlabel='Feature Importances')
    ax.set_ylabel(ylabel='Feature')

    if show:
        plt.show()

    return ax


# 규제 회귀 모델의 회귀계수 경로 시각화
def coef_path(
    X: pd.DataFrame,
    y: pd.Series,
    model: str = 'lasso',
    alphas: np.ndarray = None,
    l1_ratio: float = 0.5,
    standardize: bool = True,
    alpha: float = None,
    palette: str = 'Spectral',
    legend: bool = True,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 규제 회귀 모델(Lasso, Ridge, ElasticNet)에서 규제 강도(alpha)에
    따라 입력변수별 회귀계수가 변화하는 경로를 선 그래프로 시각화합니다.
    x축은 로그 척도이며 오른쪽으로 갈수록 규제가 강해지므로 회귀계수가 0으로
    수렴합니다.

    매개변수:
        X: 입력변수 행렬을 데이터프레임으로 지정합니다.
        y: 목표변수 벡터를 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        model: 규제 회귀 모델을 'lasso', 'ridge' 또는 'elasticnet'으로
            지정합니다.(기본값: 'lasso')
        alphas: 규제 강도를 1차원 np.ndarray로 지정합니다. 생략하면 자동으로
            생성합니다.(기본값: None)
        l1_ratio: ElasticNet의 L1 규제 비율을 0~1의 실수로 지정합니다.
            (기본값: 0.5)
        standardize: 입력변수의 표준화 여부를 True 또는 False로 지정합니다.
            (기본값: True)
        alpha: 세로 점선으로 표시할 규제 강도를 실수로 지정합니다.(기본값: None)
        palette: 팔레트를 문자열로 지정합니다.(기본값: 'Spectral')
        legend: 범례 추가 여부를 True 또는 False로 지정합니다.(기본값: True)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    if isinstance(X, pd.DataFrame):
        X_mat = X.drop(labels=['const'], axis=1, errors='ignore')
        names = list(X_mat.columns)
        X_mat = X_mat.to_numpy(dtype=float)
    else:
        X_mat = np.asarray(a=X, dtype=float)
        names = [f'X{i + 1}' for i in range(X_mat.shape[1])]

    y_vec = np.asarray(a=y, dtype=float).ravel()

    if standardize:
        X_mat = StandardScaler().fit_transform(X=X_mat)

    kind = model.lower()

    if kind in ('lasso', 'elasticnet', 'enet'):
        ratio = 1.0 if kind == 'lasso' else l1_ratio

        # 사이킷런 버전에 따라 기본 규제 강도가 달라지므로 직접 생성
        if alphas is None:
            n = X_mat.shape[0]
            xy = np.abs(X_mat.T @ (y_vec - y_vec.mean())).max()
            alpha_max = xy / (n * ratio) if xy > 0 else 1.0
            alphas = np.logspace(
                start=np.log10(alpha_max),
                stop=np.log10(alpha_max * 1e-3),
                num=100,
            )

        alphas_, coefs, _ = enet_path(
            X=X_mat,
            y=y_vec,
            alphas=alphas,
            l1_ratio=ratio,
        )
    elif kind == 'ridge':
        if alphas is None:
            alphas_ = np.logspace(start=-3, stop=3, num=100)
        else:
            alphas_ = np.asarray(a=alphas, dtype=float)
        coefs = np.array(
            object=[
                Ridge(alpha=a).fit(X=X_mat, y=y_vec).coef_ for a in alphas_
            ]
        ).T
    else:
        raise ValueError(
            "model은 'lasso', 'ridge' 또는 'elasticnet'으로 지정하세요."
        )

    colors = sns.color_palette(palette=palette, n_colors=len(names))

    for i, name in enumerate(names):
        ax.plot(alphas_, coefs[i], label=name, color=colors[i], linewidth=1)

    ax.axhline(y=0, color='0.5', linestyle='--', linewidth=0.5)

    if alpha is not None:
        ax.axvline(x=alpha, color='red', linestyle='--', linewidth=0.5)
        ax.text(
            x=alpha,
            y=ax.get_ylim()[1],
            s=f'alpha = {alpha:.4f}',
            color='red',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
        )

    ax.set_xscale('log')
    ax.set_title(
        label=f'{kind.title()} 회귀계수 경로',
        fontdict={'fontweight': 'bold'},
    )
    ax.set_xlabel(xlabel='Alpha (log scale)')
    ax.set_ylabel(ylabel='Coefficients')

    if legend is True:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        remove_legend(ax)

    if show:
        plt.show()

    return ax


# 의사결정나무 모델 가지치기 단계 그래프 시각화
def step(
    data: pd.DataFrame,
    x: str = 'alpha',
    y: str = 'impurity',
    color: str = 'blue',
    title: str = None,
    xangle: int = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 의사결정나무 모델의 사후 가지치기 결과를 단계 그래프로
    시각화합니다.

    매개변수:
        data: 의사결정나무 모델의 가지치기 단계별 비용 복잡도 파라미터를
            데이터프레임으로 지정합니다.
        x: x축에 지정할 변수명을 문자열로 지정합니다.(기본값: 'alpha')
        y: y축에 지정할 변수명을 문자열로 지정합니다.(기본값: 'impurity')
        color: 선과 점의 색을 문자열로 지정합니다.(기본값: 'blue')
        title: 그래프의 제목을 문자열로 지정합니다.(기본값: None)
        xangle: x축 회전 각도를 정수로 지정합니다.(기본값: None)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    sns.lineplot(
        data=data,
        x=x,
        y=y,
        color=color,
        drawstyle='steps-pre',
        label=y,
        ax=ax,
    )

    sns.scatterplot(data=data, x=x, y=y, color=color, s=15, ax=ax)

    if title is not None:
        ax.set_title(label=title, fontdict={'fontweight': 'bold'})

    if xangle is not None:
        ax.tick_params(axis='x', rotation=xangle)

    if show:
        plt.show()

    return ax


# 분류 모델의 ROC 곡선 시각화 및 AUC 계산 함수
def roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: str = None,
    color: str = None,
    label: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 분류 모델의 ROC 곡선을 그리고 AUC를 계산합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_prob: 목표변수의 예측 확률을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        pos_label: Positive 범주를 문자열로 지정합니다.
        color: 곡선의 색을 문자열로 지정합니다.
        label: 범례에 표시할 모델명을 문자열로 지정합니다. 생략하면 varname
            패키지가 설치된 경우 y_prob로 지정한 변수명을 사용합니다.
            (기본값: None)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    if label is None:
        label = _arg_name('y_prob')

    if isinstance(y_true, np.ndarray):
        y_class = pd.Series(data=y_true).value_counts().sort_index()
    else:
        y_class = y_true.value_counts().sort_index()

    if pos_label is None:
        pos_label = y_class.loc[y_class == y_class.min()].index[0]

    idx = np.where(y_class.index == pos_label)[0][0]

    if y_prob.ndim == 2:
        y_prob = y_prob[:, idx]

    fpr, tpr, _ = metrics.roc_curve(
        y_true=y_true,
        y_score=y_prob,
        pos_label=pos_label,
    )

    auc_value = metrics.auc(x=fpr, y=tpr)

    if label is None:
        legend = f'AUC: {auc_value:.4f}'
    else:
        legend = f'AUC({label}): {auc_value:.4f}'

    ax.plot(
        fpr,
        tpr,
        color=color,
        label=legend,
        linewidth=1.0,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color='k',
        linestyle='--',
        linewidth=0.5,
    )

    ax.set_title(label='ROC Curve', fontdict={'fontweight': 'bold'})
    ax.set_xlabel(xlabel='FPR')
    ax.set_ylabel(ylabel='TPR')
    ax.legend(loc='lower right', fontsize=8)

    if show:
        plt.show()

    return ax


# 분류 모델의 PR 곡선 시각화 및 AP 계산 함수
def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: str = None,
    color: str = None,
    label: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 분류 모델의 PR 곡선을 그리고 AP를 계산합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_prob: 목표변수의 예측 확률을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        pos_label: Positive 범주를 문자열로 지정합니다.
        color: 곡선의 색을 문자열로 지정합니다.
        label: 범례에 표시할 모델명을 문자열로 지정합니다. 생략하면 varname
            패키지가 설치된 경우 y_prob로 지정한 변수명을 사용합니다.
            (기본값: None)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    if label is None:
        label = _arg_name('y_prob')

    if isinstance(y_true, np.ndarray):
        y_class = pd.Series(data=y_true).value_counts().sort_index()
    else:
        y_class = y_true.value_counts().sort_index()

    if pos_label is None:
        pos_label = y_class.loc[y_class == y_class.min()].index[0]

    idx = np.where(y_class.index == pos_label)[0][0]

    if y_prob.ndim == 2:
        y_prob = y_prob[:, idx]

    precision, recall, _ = metrics.precision_recall_curve(
        y_true=y_true,
        y_score=y_prob,
        pos_label=pos_label,
    )

    ap = metrics.average_precision_score(
        y_true=y_true,
        y_score=y_prob,
        pos_label=pos_label,
    )

    if label is None:
        legend = f'AP: {ap:.4f}'
    else:
        legend = f'AP({label}): {ap:.4f}'

    ax.plot(
        recall,
        precision,
        color=color,
        label=legend,
        linewidth=1.0,
    )

    ax.set_title(
        label='Precision-Recall Curve',
        fontdict={'fontweight': 'bold'},
    )
    ax.set_xlabel(xlabel='Recall')
    ax.set_ylabel(ylabel='Precision')
    ax.legend(loc='lower left', fontsize=8)

    if show:
        plt.show()

    return ax


# 최적의 분류 기준점 시각화 함수
def roc_cutoff(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 분류 모델의 ROC 곡선에 최적의 분류 기준점을 추가합니다. 최적의
    분류 기준점은 민감도와 특이도의 합이 최대인 지점이며, 좌상단으로 접하는
    접선과 함께 표시합니다.

    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        y_prob: 목표변수의 예측 확률을 pd.Series 또는 1차원 np.ndarray로
            지정합니다.
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    # 순환 참조를 피하려고 함수 안에서 호출
    from hds.stat import clf_cutoffs

    ax, show = _prepare_ax(ax)

    cutoff_df = clf_cutoffs(y_true, y_prob)

    # ROC 곡선 그리기
    sns.lineplot(data=cutoff_df, x='FPR', y='TPR', color='black', ax=ax)

    # 그래프 제목 추가
    ax.set_title(
        label='최적의 분류 기준점 탐색',
        fontdict={'fontweight': 'bold'},
    )

    # 대각선 추가
    ax.plot(
        [0, 1],
        [0, 1],
        color='0.5',
        linestyle='--',
        linewidth=0.5,
    )

    # 최적의 분류 기준점 추가
    optimal = cutoff_df.iloc[[cutoff_df['Optimal'].argmax()]]

    sns.scatterplot(data=optimal, x='FPR', y='TPR', color='red', ax=ax)

    # 접선 추가
    opt_x = optimal['FPR'].iloc[0]
    opt_y = optimal['TPR'].iloc[0]

    intercept = opt_y - opt_x

    ax.plot(
        [0, 1 - intercept],
        [intercept, 1],
        color='red',
        linestyle='-.',
        linewidth=0.5,
    )

    # 분류 기준점 텍스트 추가
    ax.text(
        x=optimal['FPR'].values[0] - 0.01,
        y=optimal['TPR'].values[0] + 0.01,
        s=f"Cutoff = {optimal['Cutoff'].round(2).values[0]}",
        ha='right',
        va='bottom',
    )

    if show:
        plt.show()

    return ax


# 주성분 분석 스크리 도표 시각화
def screeplot(X: pd.DataFrame, ax: plt.Axes = None) -> plt.Axes:
    """
    이 함수는 주성분 점수 행렬을 스크리 도표로 시각화합니다.

    매개변수:
        X: 주성분 점수 행렬을 데이터프레임으로 지정합니다.
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    variances = X.var()
    n = len(variances)
    xticks = range(1, n + 1)

    sns.lineplot(
        x=xticks,
        y=variances,
        color='blue',
        linestyle='-',
        linewidth=1,
        marker='o',
        ax=ax,
    )

    ax.axhline(y=1, color='red', linestyle='--', linewidth=0.5)

    ax.set_xticks(ticks=list(xticks))
    ax.set_title(label='Scree Plot', fontdict={'fontweight': 'bold'})
    ax.set_xlabel(xlabel='Number of PC')
    ax.set_ylabel(ylabel='Variance')

    if show:
        plt.show()

    return ax


# 주성분 분석 행렬도 시각화
def biplot(
    score: pd.DataFrame,
    coefs: pd.DataFrame,
    x: int = 1,
    y: int = 2,
    zoom: float = 1.0,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    이 함수는 주성분 분석 결과를 행렬도(biplot)로 시각화합니다.

    매개변수:
        score: 주성분 점수 행렬을 데이터프레임으로 지정합니다.
        coefs: 고유벡터 행렬을 데이터프레임으로 지정합니다.
        x: x축에 지정할 주성분의 인덱스를 정수로 지정합니다.(기본값: 1)
        y: y축에 지정할 주성분의 인덱스를 정수로 지정합니다.(기본값: 2)
        zoom: 변수의 벡터 크기를 조절하는 값을 실수로 지정합니다.(기본값: 1.0)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    xs = score.iloc[:, x - 1]
    ys = score.iloc[:, y - 1]

    sns.scatterplot(
        x=xs,
        y=ys,
        fc='silver',
        ec='black',
        s=15,
        alpha=0.2,
        ax=ax,
    )

    ax.axvline(x=0, color='0.5', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='0.5', linestyle='--', linewidth=0.5)

    n = score.shape[1]

    for i in range(n):
        ax.arrow(
            x=0,
            y=0,
            dx=coefs.iloc[i, x - 1] * zoom,
            dy=coefs.iloc[i, y - 1] * zoom,
            color='red',
            linewidth=0.5,
            alpha=0.5,
        )

        ax.text(
            x=coefs.iloc[i, x - 1] * (zoom + 0.5),
            y=coefs.iloc[i, y - 1] * (zoom + 0.5),
            s=coefs.index[i],
            color='darkred',
            ha='center',
            va='center',
            fontsize=8,
            fontweight='bold',
        )

    ax.set_title(
        label='Biplot with PC1 and PC2',
        fontdict={'fontweight': 'bold'},
    )
    ax.set_xlabel(xlabel=f'PC{x}')
    ax.set_ylabel(ylabel=f'PC{y}')

    if show:
        plt.show()

    return ax


# k-평균 군집분석 WSS 단계 그래프 시각화
def wcss(X: pd.DataFrame, k: int = 3, ax: plt.Axes = None) -> plt.Axes:
    """
    이 함수는 군집별 편차 제곱합을 선 그래프로 시각화합니다.

    매개변수:
        X: 표준화된 데이터셋을 데이터프레임으로 지정합니다.
        k: 군집 개수를 정수로 지정합니다.(기본값: 3)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    ks = range(1, k + 1)
    result = []

    for n_clusters in ks:
        model = KMeans(n_clusters=n_clusters, random_state=0)
        model.fit(X=X)
        result.append(model.inertia_)

    sns.lineplot(
        x=ks,
        y=result,
        marker='o',
        linestyle='-',
        linewidth=1,
        ax=ax,
    )

    ax.set_xticks(ticks=list(ks))
    ax.set_title(label='Elbow Method', fontdict={'fontweight': 'bold'})
    ax.set_xlabel(xlabel='Number of clusters')
    ax.set_ylabel(ylabel='Within Cluster Sum of Squares')

    if show:
        plt.show()

    return ax


# k-평균 군집분석 실루엣 지수 시각화
def silhouette(X: pd.DataFrame, k: int = 3, ax: plt.Axes = None) -> plt.Axes:
    """
    이 함수는 군집별 실루엣 지수를 선 그래프로 시각화합니다.

    매개변수:
        X: 표준화된 데이터셋을 데이터프레임으로 지정합니다.
        k: 군집 개수를 정수로 지정합니다.(기본값: 3)
        ax: 그래프를 그릴 matplotlib Axes 객체를 지정합니다. 생략하면 현재
            Axes에 그린 다음 화면에 출력합니다.(기본값: None)

    반환값:
        그래프를 그린 matplotlib Axes 객체를 반환합니다.
    """
    ax, show = _prepare_ax(ax)

    ks = range(1, k + 1)
    result = [0]

    for n_clusters in ks:
        if n_clusters == 1:
            continue
        model = KMeans(n_clusters=n_clusters, random_state=0)
        model.fit(X=X)
        cluster = model.predict(X=X)
        score = metrics.silhouette_score(X=X, labels=cluster)
        result.append(score)

    sns.lineplot(
        x=ks,
        y=result,
        marker='o',
        linestyle='-',
        linewidth=1,
        ax=ax,
    )

    ax.set_xticks(ticks=list(ks))
    ax.set_title(
        label='Silhouette Width',
        fontdict={'fontweight': 'bold'},
    )
    ax.set_xlabel(xlabel='Number of clusters')
    ax.set_ylabel(ylabel='Silhouette Width Average')

    if show:
        plt.show()

    return ax


# End of Document
