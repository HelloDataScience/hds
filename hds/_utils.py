# 관련 라이브러리 호출
import functools
import importlib
import warnings
from types import ModuleType


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


# 이름을 바꾼 함수의 하위 호환 별칭을 생성하는 함수
def deprecated_alias(func, old_name: str):
    """
    이 함수는 이름을 바꾼 함수의 하위 호환 별칭을 생성합니다. 별칭을 실행하면
    새로운 함수명을 안내하는 DeprecationWarning을 발생시키고 새로운 함수를
    실행합니다.

    매개변수:
        func: 새로운 이름의 함수를 지정합니다.
        old_name: 이전 함수명을 문자열로 지정합니다.

    반환값:
        이전 함수명으로 사용할 수 있는 함수를 반환합니다.
    """
    new_name = func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            message=(
                f"'{old_name}' 함수는 '{new_name}' 함수로 이름이 "
                f"바뀌었습니다. 앞으로 '{new_name}' 함수를 사용하세요."
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    wrapper.__doc__ = (
        f"이 함수는 '{new_name}' 함수의 이전 이름입니다. "
        f"앞으로 '{new_name}' 함수를 사용하세요.\n\n{func.__doc__}"
    )
    return wrapper


# End of Document
