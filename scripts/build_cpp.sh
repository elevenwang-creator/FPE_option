#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX="${PIXI_ENV_PREFIX:-.pixi/envs/default}"

# Get version from git tag
VERSION=$(bash scripts/version.sh)

OS="$(uname)"
if [ "$OS" = "Darwin" ]; then
    SHLIB_NAME="libfpe_engine.dylib"
else
    SHLIB_NAME="libfpe_engine.so"
fi

PY_VER="$(pixi run python -c 'import sysconfig; print(sysconfig.get_config_var("py_version_short"))')"
SITE_PACKAGES="${PREFIX}/lib/python${PY_VER}/site-packages"
PY_EXT_SUFFIX="$(pixi run python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

echo "=== [1/5] Building Mojo C ABI shared library ==="
mkdir -p "${PREFIX}/lib"
mojo build -I src --emit shared-lib -o "${PREFIX}/lib/${SHLIB_NAME}" src/bindings/c_abi.mojo

if [ "$OS" = "Darwin" ]; then
    install_name_tool -id @rpath/${SHLIB_NAME} "${PREFIX}/lib/${SHLIB_NAME}"
fi

echo "=== [2/5] Installing C/C++ headers ==="
mkdir -p "${PREFIX}/include"
cp cpp/include/fpe_engine.h "${PREFIX}/include/"
cp cpp/include/fpe_compute.hpp "${PREFIX}/include/"

echo "=== [3/5] Installing Python package into site-packages ==="
mkdir -p "${SITE_PACKAGES}/fpe_engine"

echo "=== [4/5] Building Python native module ==="
mojo build -I src --emit shared-lib -o "${SITE_PACKAGES}/fpe_engine/_fpe_native${PY_EXT_SUFFIX}" src/bindings/_fpe_native.mojo
cp python/fpe_engine/__init__.py "${SITE_PACKAGES}/fpe_engine/"
cp python/fpe_engine/pricer.py "${SITE_PACKAGES}/fpe_engine/"
cat > "${SITE_PACKAGES}/fpe_engine/_version.py" <<EOF
__version__ = "${VERSION}"
EOF
echo "    Python version: ${VERSION}"

echo "=== [5/5] Building C++ examples (cmake) ==="
BUILD_DIR="cpp/examples/build"
cmake -S cpp/examples -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFPE_PREFIX="$(cd "${PREFIX}" && pwd)"
cmake --build "${BUILD_DIR}" --config Release

echo ""
echo "=== Build + install complete ==="
echo "  C/C++: #include <fpe_engine.h> → -I${PREFIX}/include -L${PREFIX}/lib -lfpe_engine"
echo "  Python: import fpe_engine → ${SITE_PACKAGES}/fpe_engine/"
echo "  C++: ${BUILD_DIR}/demo"
