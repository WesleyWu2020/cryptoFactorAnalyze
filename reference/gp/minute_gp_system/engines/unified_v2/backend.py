"""Array backend abstraction: numpy (CPU) or cupy (GPU).

Uses a dynamic proxy so that `from .backend import xp` works correctly
even after calling set_backend('gpu') — the xp object's internal module
reference is updated in-place.

Usage:
    from .backend import xp, to_numpy, set_backend
    set_backend('gpu')
    a = xp.array([1, 2, 3])
"""
import ctypes
import os
import sys

import numpy as np

_BACKEND = 'cpu'


class _XpProxy:
    """Proxy that delegates all attribute access to the current array module.

    This solves the Python import-binding issue: `from .backend import xp`
    captures a reference to THIS object, and __getattr__ always resolves
    to the current numpy/cupy module.
    """
    __slots__ = ('_mod',)

    def __init__(self, mod):
        object.__setattr__(self, '_mod', mod)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_mod'), name)

    def _set_mod(self, mod):
        object.__setattr__(self, '_mod', mod)


xp = _XpProxy(np)


def _cuda_library_dirs():
    dirs = []
    for env_name in ("GP_CUDA_LIBRARY_PATH", "CUDA_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        raw = os.environ.get(env_name, "")
        dirs.extend(part for part in raw.split(os.pathsep) if part)
    dirs.extend((
        "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.0/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
    ))
    seen = set()
    out = []
    for path in dirs:
        if path and path not in seen and os.path.isdir(path):
            seen.add(path)
            out.append(path)
    return out


def _preload_cuda_library(names):
    for directory in _cuda_library_dirs():
        for name in names:
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return path
            except OSError:
                continue
    return None


def _preload_cupy_runtime_libs():
    _preload_cuda_library(("libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"))
    _preload_cuda_library(("libcudart.so.12", "libcudart.so.13", "libcudart.so"))


def _find_cuda_runtime_dir():
    for directory in _cuda_library_dirs():
        if (
            os.path.exists(os.path.join(directory, "libnvrtc.so.12"))
            or os.path.exists(os.path.join(directory, "libnvrtc.so.13"))
        ):
            return directory
    return None


def _module_invocation_from_script(script_path):
    if not script_path or not script_path.endswith(".py"):
        return None
    path = os.path.abspath(script_path)
    if not os.path.exists(path):
        return None
    parts = [os.path.splitext(os.path.basename(path))[0]]
    directory = os.path.dirname(path)
    while os.path.exists(os.path.join(directory, "__init__.py")):
        parts.append(os.path.basename(directory))
        directory = os.path.dirname(directory)
    if len(parts) <= 1:
        return None
    return ".".join(reversed(parts))


def _ensure_cuda_runtime_on_loader_path():
    if os.environ.get("GP_DISABLE_CUDA_REEXEC", "").lower() in {"1", "true", "yes", "on"}:
        return
    if os.environ.get("_GP_CUDA_LD_REEXECED") == "1":
        return
    if sys.argv and sys.argv[0] in {"-c", "-"}:
        return
    cuda_dir = _find_cuda_runtime_dir()
    if not cuda_dir:
        return
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    if cuda_dir in current:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([cuda_dir, *current])
    env["_GP_CUDA_LD_REEXECED"] = "1"
    module_name = _module_invocation_from_script(sys.argv[0] if sys.argv else "")
    if module_name:
        args = [sys.executable, "-m", module_name, *sys.argv[1:]]
    else:
        args = [sys.executable, *sys.argv]
    os.execvpe(sys.executable, args, env)


def set_backend(backend='cpu', gpu_id=0):
    global _BACKEND
    _BACKEND = backend
    if backend == 'gpu':
        _ensure_cuda_runtime_on_loader_path()
        _preload_cupy_runtime_libs()
        import cupy as cp
        cp.cuda.Device(gpu_id).use()
        xp._set_mod(cp)
    else:
        xp._set_mod(np)


def get_backend():
    return _BACKEND


def to_numpy(arr):
    if _BACKEND == 'gpu':
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return np.asarray(arr)


def to_xp(arr):
    if _BACKEND == 'gpu':
        import cupy as cp
        return cp.asarray(arr)
    return np.asarray(arr)


def free_gpu():
    if _BACKEND == 'gpu':
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
