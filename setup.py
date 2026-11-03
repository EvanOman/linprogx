"""Build-time setup for linprogx C extensions.

When OpenBLAS is available, the sparse extension is compiled with BLAS
support for accelerated dense-tail factorization.  When it is not (e.g.
minimal deployment environments), the extension still compiles and the
solver still works -- just without BLAS-accelerated sparse internals.
"""

import subprocess
import sys

from setuptools import Extension, setup


def _have_openblas() -> bool:
    """Return True if libopenblas can be found by the linker."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import ctypes; ctypes.CDLL('libopenblas.so')"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass
    # Fallback: check if the header exists
    import os

    for path in ["/usr/include", "/usr/local/include", "/usr/include/openblas"]:
        if os.path.isfile(os.path.join(path, "cblas.h")):
            return True
    return False


have_blas = _have_openblas()

csparse_compile_args = ["-O3", "-pthread"]
csparse_link_args = ["-pthread"]
csparse_libraries: list[str] = []

if have_blas:
    csparse_compile_args.append("-DLINPROGX_HAVE_BLAS")
    csparse_libraries.append("openblas")
    print("setup.py: OpenBLAS found -- building with BLAS support")
else:
    print("setup.py: OpenBLAS not found -- building without BLAS (solver still works)")

ext_modules = [
    Extension(
        "linprogx._cfast",
        sources=["src/linprogx/_cfast.c"],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "linprogx._csparse",
        sources=["src/linprogx/_csparse.c"],
        extra_compile_args=csparse_compile_args,
        extra_link_args=csparse_link_args,
        libraries=csparse_libraries,
    ),
    # DS2 component A (CHUZC).  _ds2_chuzc.c is pure C with no CPython or
    # _csparse dependency; _ds2_chuzc_py.c is validation-harness glue.
    Extension(
        "linprogx._ds2_chuzc",
        sources=["src/linprogx/_ds2_chuzc.c", "src/linprogx/_ds2_chuzc_py.c"],
        extra_compile_args=["-O3"],
    ),
]

setup(ext_modules=ext_modules)
