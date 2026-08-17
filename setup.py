import re
from pathlib import Path

from setuptools import find_packages, setup

# 선택 의존성 패키지 목록
EXTRAS_REQUIRE = {
    # plot.tree() 함수로 의사결정나무를 시각화할 때 필요합니다.
    'tree': ['graphviz'],
    # plot.add_google_font() 함수로 구글 폰트를 설치할 때 필요합니다.
    'font': ['requests', 'beautifulsoup4'],
    # stat.clf_metrics() 함수의 결과를 주피터에서 나란히 출력할 때 필요합니다.
    'notebook': ['ipywidgets', 'ipython'],
    # plot.roc_curve()와 plot.pr_curve() 함수에서 범례에 변수명을 자동으로
    # 표시할 때 필요합니다.(label 매개변수로 대체할 수 있습니다.)
    'varname': ['varname'],
}

EXTRAS_REQUIRE['all'] = sorted(
    {package for packages in EXTRAS_REQUIRE.values() for package in packages}
)


def read_version():
    init_file = Path(__file__).parent / 'hds' / '__init__.py'
    text = init_file.read_text(encoding='UTF-8')
    match = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", text, re.M)
    if match is None:
        raise RuntimeError('Unable to find __version__ in hds/__init__.py')
    return match.group(1)


with open('README.md', encoding='UTF-8') as file:
    long_description = file.read()

setup(
    name='hds',
    version=read_version(),
    author='HelloDataScience',
    author_email='hellodatasciencekorea@gmail.com',
    description='Functions for EDA, Statistics and Machine Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HelloDataScience/hds',
    project_urls={
        'Bug Tracker': 'https://github.com/HelloDataScience/hds/issues',
    },
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'seaborn>=0.13',
        'matplotlib',
        'statsmodels',
        'scikit-learn>=1.4',
    ],
    extras_require=EXTRAS_REQUIRE,
)
