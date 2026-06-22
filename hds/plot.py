# 관련 라이브러리 호출
import requests
from bs4 import BeautifulSoup as bts
import json
import os
import re
import platform
import shutil
import subprocess
import matplotlib
import glob
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import inspect
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.cluster import KMeans
from sklearn import metrics
from varname import argname


# 구글 폰트 파일 목록을 반환하는 함수
def search_google_font_file(font_name: str) -> list:
    '''
    이 함수는 구글 폰트(https://fonts.google.com)에 등록된 폰트명을 지정하면
    해당 폰트명의 ttf 파일 목록을 반환합니다.
    
    매개변수:
        font_name: 구글 폰트명을 문자열로 지정합니다.
    
    반환값:
        구글 폰트 ttf 파일명을 리스트로 반환합니다.
    '''
    # 구글 폰트명에서 공백 제거
    font_name_removed = font_name.replace(' ', '')
    
    # 구글 폰트명 URL 생성
    url = f'https://github.com/google/fonts/tree/main/ofl/{font_name_removed.lower()}'
    
    # 구글 폰트 파일 목록 내려받기
    res = requests.get(url)
    if res.status_code == 200:
        soup = bts(markup = res.text, features = 'html.parser')
        items = soup.select('script[type="application/json"][data-target="react-app.embeddedData"]')
        dat = json.loads(s = items[0].text)
        files = dat['payload']['tree']['items']
        return [file['name'] for file in files if '.ttf' in file['name']]
    else:
        raise FileNotFoundError(f'Font not found with {font_name}')


# 구글 폰트 파일을 다운로드 폴더에 내려받는 함수
def download_google_font_file(font_file: str) -> str:
    '''
    이 함수는 구글 폰트 ttf 파일명을 지정하면 다운로드 폴더에 내려받습니다.
    
    매개변수:
        font_file: 구글 폰트 ttf 파일명을 문자열로 지정합니다.
    
    반환값:
        다운로드 폴더에 내려받은 구글 폰트 ttf 파일명을 문자열로 반환합니다.
    '''
    # 다운로드 폴더 경로 지정
    download_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    os.makedirs(name = download_path, exist_ok = True)
    font_path = os.path.join(download_path, font_file)
    
    # 구글 폰트명 생성
    font_name = re.split(pattern = r'(-)|(\[)|(\.ttf)', string = font_file)[0].lower()
    
    # 구글 폰트 파일 다운로드 URL 생성
    domain = 'https://raw.githubusercontent.com/google/fonts/refs/heads/main/ofl/'
    url = os.path.join(domain, font_name, font_file)
    
    # 구글 폰트 파일 내려받기
    res = requests.get(url)
    if res.status_code == 200:
        with open(file = font_path, mode = 'wb') as file:
            file.write(res.content)
        print(f'Downloaded to {font_path}')
        return font_path
    else:
        raise FileNotFoundError(f'Font not found at {url}')


# 구글 폰트를 설치하고 다운로드 폴더에서 삭제하는 함수
def install_google_font_path(font_path: str) -> None:
    '''
    이 함수는 다운로드 폴더에 내려받은 구글 폰트 ttf 파일명을 운영체제에 맞게 설치하고
    다운로드 폴더에 있는 ttf 파일명을 삭제합니다.
    
    매개변수:
        font_path: 다운로드 폴더에 내려받은 구글 폰트 ttf 파일명을 문자열로 지정합니다.
    
    반환값:
        없습니다.
    '''
    # 운영체제별 구글 폰트 설치 경로 지정
    system = platform.system()
    if system == 'Windows':
        fonts_dir = os.path.join(os.getenv(key = 'WINDIR'), 'Fonts')
        shutil.copy(src = font_path, dst = fonts_dir)
    elif system == 'Darwin':
        fonts_dir = os.path.expanduser('~/Library/Fonts')
        shutil.copy(src = font_path, dst = fonts_dir)
    elif system == 'Linux':
        fonts_dir = os.path.expanduser('~/.fonts')
        os.makedirs(name = fonts_dir, exist_ok = True)
        shutil.copy(src = font_path, dst = fonts_dir)
        subprocess.run(['fc-cache', '-f', '-v'])
    else:
        raise OSError('Unsupported operating system')
    
    # 실행 완료 문구 출력
    print(f'Installed font at {fonts_dir}')
    
    # 구글 폰트 파일 삭제
    os.remove(font_path)


# 구글 폰트를 설치하고 matplotlib 임시 폴더에 있는 json 파일을 삭제하는 함수
def add_google_font(font_name: str) -> None:
    '''
    이 함수는 구글 폰트명을 지정하면 해당 폰트의 ttf 파일명을 다운로드 폴더에 
    내려받은 다음 운영체제에 맞게 설치하고 다운로드 폴더에서 삭제합니다.
    
    매개변수:
        font_name: 구글 폰트명을 문자열로 지정합니다.
    
    반환값:
        없습니다.
    '''
    # 구글 폰트 파일 목록 생성
    font_files = search_google_font_file(font_name)
    
    # 반복문 실행
    for font_file in font_files:
        try:
            # 구글 폰트 파일을 다운로드 폴더에 내려받기
            font_path = download_google_font_file(font_file)
            
            # 구글 폰트를 설치하고 다운로드 폴더에서 삭제
            install_google_font_path(font_path)
        
        except Exception as e:
            print(f'Error: {e}')
    
    # matplotlib 임시 폴더에 있는 json 파일 삭제
    path = matplotlib.get_cachedir()
    file = glob.glob(f'{path}/fontlist-*.json')[0]
    os.remove(path = file)


# 범례가 있을 때만 제거하도록 변경
def remove_legend():
    legend = plt.gca().get_legend()
    if legend is not None:
        legend.remove()

# 집단별 상자 수염 그림을 그리는 함수
def box_group(data: pd.DataFrame, x: str, y: str, palette: list = None, legend: bool = False) -> None:
    '''
    이 함수는 범주형 변수(x축)에 따라 연속형 변수(y축)의 상자 수염 그림을 그립니다.
    상자에 빨간 점은 해당 범주의 평균이며, 가로 직선은 전체 평균입니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        y: 연속형 변수명을 문자열로 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.
        legend: 범례 추가 여부를 True 또는 False로 지정합니다.(기본값: False)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    avg = data.groupby(by = x)[y].mean()
    
    sns.boxplot(
        data = data, 
        x = x, 
        y = y, 
        hue = x, 
        order = avg.index, 
        palette = palette, 
        flierprops = {
            'marker': 'o', 
            'markersize': 3, 
            'markerfacecolor': 'pink',
            'markeredgecolor': 'red', 
            'markeredgewidth': 0.2
        }, 
        linecolor = '0.5',
        linewidth = 0.5
    )
    
    plt.axhline(
        y = data[y].mean(), 
        color = 'red', 
        linewidth = 0.5, 
        linestyle = '--'
    )
    
    for i, v in enumerate(avg):
        plt.text(
            x = i, 
            y = v, 
            s = f'{v:,.2f}', 
            ha = 'center', 
            va = 'center',
            fontsize = 6, 
            fontweight = 'bold'
        )
    
    if legend is True:
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), title = x)
    else:
        remove_legend()
    
    plt.title(label = f'{x} 범주별 {y}의 평균 비교', fontdict = {'fontweight': 'bold'});


# 두 연속형 변수로 산점도를 그리는 함수
def scatter(data: pd.DataFrame, x: str, y: str, color: str = '0.3') -> None:
    '''
    이 함수는 두 연속형 변수의 산점도를 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 원인이 되는 연속형 변수명을 문자열로 지정합니다.
        y: 결과가 되는 연속형 변수명을 문자열로 지정합니다.
        color: 점의 채우기 색을 문자열로 지정합니다.(기본값: '0.3')
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    sns.scatterplot(
        data = data, 
        x = x, 
        y = y, 
        color = color
    )
    
    plt.title(label = f'{x}와(과) {y}의 관계', fontdict = {'fontweight': 'bold'});


# 두 연속형 변수로 산점도와 회귀직선을 그리는 함수
def regline(data: pd.DataFrame, x: str, y: str, color: str = '0.3', size: int = 15) -> None:
    '''
    이 함수는 두 연속형 변수의 산점도에 회귀직선을 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 원인이 되는 연속형 변수명을 문자열로 지정합니다.
        y: 결과가 되는 연속형 변수명을 문자열로 지정합니다.
        color: 점의 채우기 색을 문자열로 지정합니다.(기본값: '0.3')
        size: 점의 크기를 정수로 지정합니다.(기본값: 15)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    sns.regplot(
        data = data, 
        x = x, 
        y = y, 
        ci = None, 
        scatter_kws = {
            'facecolor': color, 
            'edgecolor': '1', 
            's': size, 
            'alpha': 0.2
        },
        line_kws = {
            'color': 'red', 
            'linewidth': 1.5
        }
    )
    
    plt.title(label = f'{x}와(과) {y}의 관계', fontdict = {'fontweight': 'bold'});


# 범주형 변수의 도수로 막대 그래프를 그리는 함수
def bar_freq(data: pd.DataFrame, x: str, color: str = None, palette: list = None, legend: bool = False) -> None:
    '''
    이 함수는 범주형 변수의 도수를 내림차순 정렬한 막대 그래프를 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        color: 점의 채우기 색을 문자열로 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.
        legend: 범례 추가 여부를 True 또는 False로 지정합니다.(기본값: False)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    grp = data[x].value_counts().sort_index()
    v_max = grp.values.max()
    space = np.ceil(v_max * 0.01)
    
    sns.countplot(
        data = data, 
        x = x, 
        hue = x, 
        order = grp.index, 
        color = color, 
        palette = palette,
    )
    
    for i, v in enumerate(grp):
        plt.text(
            x = i, 
            y = v + space, 
            s = v, 
            ha = 'center', 
            va = 'bottom', 
            c = 'black', 
            fontsize = 8, 
            fontweight = 'bold'
        )

    if legend is True:
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), title = x)
    else:
        remove_legend()
    
    plt.ylim(0, v_max * 1.2)
    plt.title(label = '목표변수의 범주별 도수 비교', fontdict = {'fontweight': 'bold'});


# 범주형 변수를 소그룹으로 나누고 도수로 펼친 막대 그래프를 그리는 함수
def bar_dodge_freq(data: pd.DataFrame, x: str, g: str, palette: list = None) -> None:
    '''
    이 함수는 범주형 변수를 소그룹으로 나누고 도수로 펼친 막대 그래프를 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        g: x를 소그룹으로 나눌 범주형 변수명을 문자열로 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    grp = data.groupby(by = [x, g]).count().iloc[:, 0]
    v_max = grp.values.max()
    space = np.ceil(v_max * 0.01)
    
    sns.countplot(
        data = data, 
        x = x, 
        hue = g, 
        order = grp.index.levels[0], 
        hue_order = grp.index.levels[1], 
        palette = palette,
    )
    
    for i, v in enumerate(grp):
        if i % 2 == 0:
            i = i/2 - 0.2
        else:
            i = (i-1)/2 + 0.2
        plt.text(
            x = i, 
            y = v + space, 
            s = v, 
            ha = 'center', 
            va = 'bottom', 
            fontsize = 8, 
            fontweight = 'bold'
        )
    
    plt.ylim(0, v_max * 1.2)
    plt.title(label = f'{x}의 범주별 {g}의 도수 비교', fontdict = {'fontweight': 'bold'})
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fontsize = 8);


# 범주형 변수를 소그룹으로 나누고 도수로 쌓은 막대 그래프를 그리는 함수
def bar_stack_freq(data: pd.DataFrame, x: str, g: str, kind: str = 'bar', palette: list = None) -> None:
    '''
    이 함수는 범주형 변수를 소그룹으로 나누고 도수로 쌓은 막대 그래프를 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        g: x를 소그룹으로 나눌 범주형 변수명을 문자열로 지정합니다.
        kind: 막대 그래프의 종류를 'bar' 또는 'barh'로 지정합니다.(기본값: 'bar')
        palette: 팔레트를 리스트로 지정합니다.
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    p = data[g].unique().size
    
    pv = pd.pivot_table(
        data = data, 
        index = x, 
        columns = g, 
        aggfunc = 'count'
    )
    
    pv = pv.iloc[:, 0:p].sort_index()
    pv.columns = pv.columns.droplevel(level = 0)
    pv.columns.name = None
    pv = pv.reset_index()
    cols = pv.columns[1:]
    cumsum = pv[cols].cumsum(axis = 1)
    
    if type(palette) == list:
        palette = sns.set_palette(sns.color_palette(palette))
    
    pv.plot(
        x = x, 
        kind = kind, 
        stacked = True, 
        rot = 0, 
        legend = 'reverse', 
        colormap = palette
    )
    
    if kind == 'bar':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                plt.text(
                    x = i, 
                    y = v1 - v2/2, 
                    s = v2, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black', 
                    fontsize = 8, 
                    fontweight = 'bold'
                )
    elif kind == 'barh':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                plt.text(
                    x = v1 - v2/2, 
                    y = i, 
                    s = v2, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black', 
                    fontsize = 8, 
                    fontweight = 'bold'
                )

    plt.title(label = f'{x}의 범주별 {g}의 도수 비교', fontweight = 'bold')
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fontsize = 8);


# 범주형 변수를 소그룹으로 나누고 상대도수로 쌓은 막대 그래프를 그리는 함수
def bar_stack_prop(data: pd.DataFrame, x: str, g: str, kind: str = 'bar', palette: list = None) -> None:
    '''
    이 함수는 범주형 변수를 소그룹으로 나누고 상대도수로 쌓은 막대 그래프를 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: 범주형 변수명을 문자열로 지정합니다.
        g: x를 소그룹으로 나눌 범주형 변수명을 문자열로 지정합니다.
        kind: 막대 그래프의 종류를 'bar' 또는 'barh'로 지정합니다.(기본값: 'bar')
        palette: 팔레트를 리스트로 지정합니다.(기본값: None)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    p = data[g].unique().size
    
    pv = pd.pivot_table(
        data = data, 
        index = x, 
        columns = g, 
        aggfunc = 'count'
    )
    
    pv = pv.iloc[:, 0:p].sort_index()
    pv.columns = pv.columns.droplevel(level = 0)
    pv.columns.name = None
    pv = pv.reset_index()
    cols = pv.columns[1:]
    rowsum = pv[cols].apply(func = sum, axis = 1)
    pv[cols] = pv[cols].div(rowsum, 0) * 100
    cumsum = pv[cols].cumsum(axis = 1)
    
    if type(palette) == list:
        palette = sns.set_palette(sns.color_palette(palette))
        
    pv.plot(
        x = x, 
        kind = kind, 
        stacked = True, 
        rot = 0, 
        legend = 'reverse', 
        colormap = palette, 
        mark_right = True
    )
    
    if kind == 'bar':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                v3 = f'{np.round(v2, 1)}%'
                plt.text(
                    x = i, 
                    y = v1 - v2/2, 
                    s = v3, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black', 
                    fontsize = 8, 
                    fontweight = 'bold'
                )
    elif kind == 'barh':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                v3 = f'{np.round(v2, 1)}%'
                plt.text(
                    x = v1 - v2/2, 
                    y = i, 
                    s = v3, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black', 
                    fontsize = 8, 
                    fontweight = 'bold'
                )
    
    plt.title(label = f'{x}의 범주별 {g}의 상대도수 비교', fontweight = 'bold')
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), fontsize = 8);


# 연속형 변수 간 상관관계 히트맵 시각화
def corr_heatmap(data: pd.DataFrame, palette: str = 'RdYlBu', fontsize = 8) -> None:
    '''
    이 함수는 연속형 변수 간 상관관계를 히트맵으로 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        palette: 팔레트를 리스트로 지정합니다.(기본값: 'RdYlBu')
        fontsize: 주석(상관계수)의 글자 크기를 지정합니다.(기본값: 8)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    corr = data.corr(numeric_only = True)
    mask = np.triu(m = np.ones_like(a = corr, dtype = bool), k = 1)
    
    sns.heatmap(data = corr, 
                cmap = palette, 
                annot = True, 
                fmt = '.2f', 
                annot_kws = {'fontweight': 'bold', 'fontsize': fontsize}, 
                linewidth = 1, 
                mask = mask)
    
    plt.title(label = '변수 간 상관계수 행렬', fontdict = {'fontweight': 'bold'});


# 등고선을 추가한 이차원 커널 밀도 곡선 시각화
def kde2d(data: pd.DataFrame, x: str, y: str, frac: float = 0.2, seed: int = 0, scatter: bool = False):
    '''
    이 함수는 범주형 변수를 소그룹으로 나누고 상대도수로 쌓은 막대 그래프를 그립니다.
    
    매개변수:
        data: 데이터프레임을 지정합니다.
        x: x축에 놓을 변수명을 문자열로 지정합니다.
        y: y축에 놓을 변수명을 문자열로 지정합니다.
        frac: 산점도를 그릴 샘플 비율을 0~1의 실수로 지정합니다.(기본값: 0.2)
        seed: 시드 초기값을 정수로 지정합니다.(기본값: 0)
        scatter: 산점도 추가 여부를 True 또는 False로 지정합니다.(기본값: False)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    levels = np.arange(start = 0.05, stop = 1.05, step = 0.05)
    sns.kdeplot(data = data, x = x, y = y, color = '0.9', fill = True, levels = levels)
    if scatter:
        data_sample = data.sample(frac = frac, random_state = seed)
        sns.scatterplot(data = data_sample, x = x, y = y, c = '0', s = 10, alpha = 0.2)
    plt.title(label = f'{x}와 {y}의 관계', fontdict = {'fontweight': 'bold'});


# 의사결정나무 모델 시각화
def tree(model, fileName: str = None, className: str = None) -> None:
    '''
    이 함수는 의사결정나무 모델을 시각화하여 png 파일로 저장합니다.
    
    매개변수:
        model: 사이킷런으로 적합한 의사결정나무 모델을 지정합니다.
        fileName: 입력변수명을 문자열로 지정합니다.(기본값: None)
        className: 분류 모델은 목표변수의 범주를 문자열로 지정합니다.(기본값: None)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    if fileName == None:
        global_objs = inspect.currentframe().f_back.f_globals.items()
        result = [name for name, value in global_objs if value is model]
        fileName = result[0]
    
    if type(model) == DecisionTreeRegressor:
        export_graphviz(
            decision_tree = model,
            out_file = f'{fileName}.dot',
            feature_names = model.feature_names_in_,
            filled = True,
            leaves_parallel = False,
            impurity = True
        )
    elif type(model) == DecisionTreeClassifier:
        if className == None:
            className = model.classes_.astype(str)
        export_graphviz(
            decision_tree = model,
            out_file = f'{fileName}.dot',
            class_names = className,
            feature_names = model.feature_names_in_,
            filled = True,
            leaves_parallel = False,
            impurity = True
        )
    
    with open(file = f'{fileName}.dot', mode = 'rt') as file:
        graph = file.read()
        graph = graphviz.Source(source = graph, format = 'png')
        graph.render(filename = fileName)
    
    os.remove(f'{fileName}')
    os.remove(f'{fileName}.dot')


# 입력변수별 중요도 시각화
def feature_importance(model, palette: str = 'Spectral') -> None:
    '''
    이 함수는 입력변수별 중요도를 막대 그래프로 시각화합니다.
    
    매개변수:
        model: 사이킷런으로 적합한 분류 모델을 지정합니다.
        palette: 팔레트를 문자열로 지정합니다.(기본값: Spectral)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    if 'LGBM' in str(type(model)):
        fi = pd.DataFrame(
            data = model.feature_importances_, 
            index = model.feature_name_, 
            columns = ['importance']
        )
    else:
        fi = pd.DataFrame(
            data = model.feature_importances_, 
            index = model.feature_names_in_, 
            columns = ['importance']
        ) \
        .sort_values(by = 'importance', ascending = False) \
        .reset_index()
    
    sns.barplot(
        data = fi, 
        x = 'importance', 
        y = 'index', 
        hue = 'index', 
        palette = palette, 
        # legend = True
    )
    
    for i, r in fi.iterrows():
        plt.text(
            x = r['importance'] + 0.01, 
            y = i, 
            s = f"{r['importance']:.3f}", 
            ha = 'left', 
            va = 'center', 
            fontsize = 8, 
            fontweight = 'bold'
        )
    
    plt.xlim(0, fi['importance'].max() * 1.2)
    plt.title(label = 'Feature Importances', fontdict = {'fontweight': 'bold'})
    plt.xlabel(xlabel = 'Feature Importances')
    plt.ylabel(ylabel = 'Feature');


# 의사결정나무 모델 가지치기 단계 그래프 시각화
def step(data: pd.DataFrame, x: str = 'alpha', y: str = 'impurity', color: str = 'blue', title: str = None, xangle: int = None) -> None:
    '''
    이 함수는 의사결정나무 모델의 사후 가지치기 결과를 단계 그래프로 시각화합니다.
    
    매개변수:
        data: 의사결정나무 모델의 가지치기 단계별 비용 복잡도 파라미터를 데이터프레임으로 지정합니다.
        x: x축에 지정할 변수명을 문자열로 지정합니다.(기본값: 'alpha')
        y: y축에 지정할 변수명을 문자열로 지정합니다.(기본값: 'impurity')
        color: 선과 점의 색을 문자열로 지정합니다.(기본값: 'blue')
        title: 그래프의 제목을 문자열로 지정합니다.(기본값: None)
        xangle: x축 회전 각도를 정수로 지정합니다.(기본값: None)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    sns.lineplot(
        data = data, 
        x = x, 
        y = y, 
        color = color, 
        drawstyle = 'steps-pre', 
        label = y
    )
    
    sns.scatterplot(
        data = data, 
        x = x, 
        y = y, 
        color = color, 
        s = 15
    )

    if title != None:
        plt.title(label = title, fontweight = 'bold')

    plt.xticks(rotation = xangle);


# 분류 모델의 ROC 곡선 시각화 및 AUC 계산 함수
def roc_curve(y_true: np.ndarray, y_prob: np.ndarray, pos_label: str = None, color: str = None) -> None:
    '''
    이 함수는 분류 모델의 ROC 곡선을 그리고 AUC를 계산합니다.
    
    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_prob: 목표변수의 예측 확률을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        pos_label: Positive 범주를 문자열로 지정합니다.
        color: 곡선의 색을 문자열로 지정합니다.
    
    반환:
        ROC 곡선 그래프 외에 반환하는 객체는 없습니다.
    '''
    if isinstance(y_true, np.ndarray):
        y_class = pd.Series(data = y_true).value_counts().sort_index()
    else:
        y_class = y_true.value_counts().sort_index()
    
    if pos_label == None:
        pos_label = y_class.loc[y_class == y_class.min()].index[0]
    
    idx = np.where(y_class.index == pos_label)[0][0]
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, idx]
        
    fpr, tpr, _ = metrics.roc_curve(
        y_true = y_true, 
        y_score = y_prob, 
        pos_label = pos_label
    )
    
    auc_ = metrics.auc(x = fpr, y = tpr)
    
    var = argname(arg = 'y_prob')
    
    plt.plot(
        fpr, 
        tpr, 
        color = color, 
        label = f'AUC({var}): {auc_:.4f}', 
        linewidth = 1.0
    )
    
    plt.plot(
        [0, 1], 
        [0, 1], 
        color = 'k', 
        linestyle = '--', 
        linewidth = 0.5
    )
    
    plt.title(label = 'ROC Curve', fontdict = {'fontweight': 'bold'})
    plt.xlabel(xlabel = 'FPR')
    plt.ylabel(ylabel = 'TPR')
    plt.legend(loc = 'lower right', fontsize = 8);


# 분류 모델의 PR 곡선 시각화 및 AP 계산 함수
def pr_curve(y_true: np.ndarray, y_prob: np.ndarray, pos_label: str = None, color: str = None) -> None:
    '''
    이 함수는 분류 모델의 PR 곡선을 그리고 AP를 계산합니다.
    
    매개변수:
        y_true: 목표변수의 실제값을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        y_prob: 목표변수의 예측 확률을 pd.Series 또는 1차원 np.ndarray로 지정합니다.
        pos_label: Positive 범주를 문자열로 지정합니다.
        color: 곡선의 색을 문자열로 지정합니다.
    
    반환:
        PR 곡선 그래프 외에 반환하는 객체는 없습니다.
    '''
    if isinstance(y_true, np.ndarray):
        y_class = pd.Series(data = y_true).value_counts().sort_index()
    else:
        y_class = y_true.value_counts().sort_index()
    
    if pos_label is None:
        pos_label = y_class.loc[y_class == y_class.min()].index[0]
    
    idx = np.where(y_class.index == pos_label)[0][0]
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, idx]
    
    precision, recall, _ = metrics.precision_recall_curve(
        y_true = y_true, 
        y_score = y_prob, 
        pos_label = pos_label
    )

    ap = metrics.average_precision_score(
        y_true = y_true, 
        y_score = y_prob, 
        pos_label = pos_label
    )

    var = argname(arg = 'y_prob')
    
    plt.plot(
        recall, 
        precision, 
        color = color, 
        label = f'AP({var}): {ap:.4f}', 
        linewidth = 1.0
    )
    
    plt.title(label = 'Precision-Recall Curve', fontdict = {'fontweight': 'bold'})
    plt.xlabel(xlabel = 'Recall')
    plt.ylabel(ylabel = 'Precision')
    plt.legend(loc = 'lower left', fontsize = 8);


# 주성분 분석 스크리 도표 시각화
def screeplot(X: pd.DataFrame) -> None:
    '''
    이 함수는 주성분 점수 행렬을 스크리 도표로 시각화합니다.
    
    매개변수:
        X: 주성분 점수 행렬을 데이터프레임으로 지정합니다.
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    X = X.var()
    n = len(X)
    xticks = range(1, n + 1)
    
    sns.lineplot(
        x = xticks, 
        y = X, 
        color = 'blue',
        linestyle = '-', 
        linewidth = 1, 
        marker = 'o'
    )
    
    plt.axhline(
        y = 1, 
        color = 'red', 
        linestyle = '--', 
        linewidth = 0.5
    )
    
    plt.xticks(ticks = xticks)
    plt.title(label = 'Scree Plot', fontdict = {'fontweight': 'bold'})
    plt.xlabel(xlabel = 'Number of PC')
    plt.ylabel(ylabel = 'Variance');
    

# 주성분 분석 행렬도 시각화
def biplot(score: pd.DataFrame, coefs: pd.DataFrame, x: int = 1, y: int = 2, zoom: float = 1.0) -> None:
    '''
    이 함수는 주성분 분석 결과를 스크리 도표로 시각화합니다.
    
    매개변수:
        score: 주성분 점수 행렬을 데이터프레임으로 지정합니다.
        coefs: 고유벡터 행렬을 데이터프레임으로 지정합니다.
        x: x축에 지정할 주성분의 인덱스를 정수로 지정합니다.(기본값: 1)
        y: y축에 지정할 주성분의 인덱스를 정수로 지정합니다.(기본값: 2)
        zoom: 변수의 벡터 크기를 조절하는 값을 실수로 지정합니다. (기본값: 1.0)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    xs = score.iloc[:, x-1]
    ys = score.iloc[:, y-1]
    
    sns.scatterplot(
        x = xs, 
        y = ys, 
        fc = 'silver',
        ec = 'black',
        s = 15, 
        alpha = 0.2
    )
    
    plt.axvline(
        x = 0, 
        color = '0.5', 
        linestyle = '--', 
        linewidth = 0.5
    )
    
    plt.axhline(
        y = 0, 
        color = '0.5', 
        linestyle = '--', 
        linewidth = 0.5
    )
    
    n = score.shape[1]
    
    for i in range(n):
        plt.arrow(
            x = 0, 
            y = 0, 
            dx = coefs.iloc[i, x-1] * zoom, 
            dy = coefs.iloc[i, y-1] * zoom, 
            color = 'red',
            linewidth = 0.5,
            alpha = 0.5
        )
        
        plt.text(
            x = coefs.iloc[i, x-1] * (zoom + 0.5), 
            y = coefs.iloc[i, y-1] * (zoom + 0.5), 
            s = coefs.index[i], 
            color = 'darkred', 
            ha = 'center', 
            va = 'center', 
            fontsize = 8, 
            fontweight = 'bold'
        )
    
    plt.title(label = 'Biplot with PC1 and PC2', fontdict = {'fontweight': 'bold'})
    plt.xlabel(xlabel = 'PC{}'.format(x))
    plt.ylabel(ylabel = 'PC{}'.format(y));


# k-평균 군집분석 WSS 단계 그래프 시각화
def wcss(X: pd.DataFrame, k: int = 3) -> None:
    '''
    이 함수는 군집별 편차 제곱합을 선 그래프로 시각화합니다.
    
    매개변수:
        X: 표준화된 데이터셋을 데이터프레임으로 지정합니다.
        k: 군집 개수를 정수로 지정합니다.(기본값: 3)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    ks = range(1, k + 1)
    result = []
    
    for k in ks:
        model = KMeans(n_clusters = k, random_state = 0)
        model.fit(X = X)
        wcss = model.inertia_
        result.append(wcss)
    
    sns.lineplot(
        x = ks, 
        y = result, 
        marker = 'o', 
        linestyle = '-', 
        linewidth = 1
    )
    
    plt.xticks(ticks = ks)
    plt.title(label = 'Elbow Method', fontdict = {'fontweight': 'bold'})
    plt.xlabel(xlabel = 'Number of clusters')
    plt.ylabel(ylabel = 'Within Cluster Sum of Squares');


# k-평균 군집분석 실루엣 지수 시각화
def silhouette(X: pd.DataFrame, k: int = 3) -> None:
    '''
    이 함수는 군집별 실루엣 지수를 선 그래프로 시각화합니다.
    
    매개변수:
        X: 표준화된 데이터셋을 데이터프레임으로 지정합니다.
        k: 군집 개수를 정수로 지정합니다.(기본값: 3)
    
    반환값:
        그래프 외에 반환하는 객체는 없습니다.
    '''
    ks = range(1, k + 1)
    result = [0]
    
    for k in ks:
        if k == 1: continue
        model = KMeans(n_clusters = k, random_state = 0)
        model.fit(X = X)
        cluster = model.predict(X = X)
        silwidth = metrics.silhouette_score(X = X, labels = cluster)
        result.append(silwidth)
    
    sns.lineplot(
        x = ks, 
        y = result, 
        marker = 'o', 
        linestyle = '-', 
        linewidth = 1
    )
    
    plt.xticks(ticks = ks)
    plt.title(label = 'Silhouette Width', fontdict = {'fontweight': 'bold'})
    plt.xlabel(xlabel = 'Number of clusters')
    plt.ylabel(ylabel = 'Silhouette Width Average');


## End of Document