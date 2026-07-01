from contextlib import contextmanager
from functools import wraps
from pylamp._C import set_capture_enabled, is_capture_enabled


@contextmanager
def capture_on(enable: bool = True):
    prev = is_capture_enabled()
    try:
        set_capture_enabled(enable)
        yield
    finally:
        set_capture_enabled(prev)


def capture(enable: bool = True):
    def decorator(fn):
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            prev = is_capture_enabled()
            set_capture_enabled(enable)

            try:
                return fn(*args, **kwargs)
            finally:
                set_capture_enabled(prev)

        return _wrapper

    return decorator
