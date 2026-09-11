# 공개 이름 목록(__all__) 테스트
import ast
from pathlib import Path

import pytest

from hds import plot, stat

MODULES = {'plot': plot, 'stat': stat}


# 모듈 소스에서 밑줄로 시작하지 않는 함수와 클래스 이름을 정의 순서대로 반환
def defined_names(module):
    source = Path(module.__file__).read_text(encoding='utf-8')
    return [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and not node.name.startswith('_')
    ]


@pytest.mark.parametrize('name', sorted(MODULES))
def test_all_matches_defined_names(name):
    module = MODULES[name]
    assert module.__all__ == defined_names(module)


@pytest.mark.parametrize('name', sorted(MODULES))
def test_all_names_exist(name):
    module = MODULES[name]
    for item in module.__all__:
        assert hasattr(module, item)


@pytest.mark.parametrize('name', sorted(MODULES))
def test_dir_shows_only_public_names(name):
    module = MODULES[name]
    assert dir(module) == sorted(module.__all__)
    for imported in ('np', 'pd', 'plt', 'metrics', 'stats'):
        assert imported not in dir(module)


def test_statsmodels_base_class_is_hidden():
    assert 'RegressionModel' not in dir(stat)
    assert 'RegressionModel' not in stat.__all__

    # 목록에서 감췄을 뿐 내부 동작에는 영향이 없음
    assert stat.RegressionModel.__module__.startswith('statsmodels')


def test_package_exposes_modules():
    import hds

    assert hds.__all__ == ['plot', 'stat']
    assert hds.plot is plot and hds.stat is stat
