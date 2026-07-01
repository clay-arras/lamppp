from contextlib import contextmanager
from pylamp._C import set_capture_enabled, is_capture_enabled


@contextmanager
def capture_on(enable: bool = True):
    prev = is_capture_enabled()
    try:
        set_capture_enabled(enable)
        yield
    finally:
        set_capture_enabled(prev)


def capture(fn):
    def _wrapper(enable: bool = True):
        prev = is_capture_enabled()
        set_capture_enabled(enable)
        out = fn()

        set_capture_enabled(prev)
        return out

    return _wrapper
