# FPE Engine: pixi-build/rattler-build + GitHub Actions Design

**Goal:** Migrate FPE Engine from `scripts/build_cpp.sh` to pixi-build/rattler-build for reproducible conda package builds, and set up GitHub Actions CI/CD for publishing to prefix.dev.

## Architecture

The project has two distinct build outputs that require a custom build script:

1. **Mojo C ABI shared library** — `libfpe_engine.{dylib,so}` + C header `fpe_engine.h`
2. **Python native extension** — `_fpe_native` + Python package files (`__init__.py`, `pricer.py`)

The `pixi-build-mojo` backend only supports pure Mojo packages (auto-derivation from `__init__.mojo`). It cannot handle our mixed-output build. We use `pixi-build-rattler-build` with a custom `recipe.yaml` that replicates the `build_cpp.sh` logic using rattler-build's environment variables (`$PREFIX`, `$SRC_DIR`).

## Design Decisions

- **Backend:** `pixi-build-rattler-build` (not `pixi-build-mojo`) — required for custom build script
- **Publishing target:** prefix.dev only (`https://prefix.dev/fpe-engine`)
- **Trusted publishing:** OIDC via GitHub Actions — no API keys stored as secrets
- **Backward compat:** `scripts/build_cpp.sh` retained for local dev, not removed
- **Platforms:** `osx-arm64`, `linux-64` (matching current `pixi.toml`)
- **Build number strategy:** Bump `number` on rebuilds of the same version; reset to 0 on version changes

## Step 1: pixi.toml Changes

Add to existing `pixi.toml`:

```toml
[workspace]
# existing fields unchanged
preview = ["pixi-build"]

[package]
name = "fpe-engine"
version = "0.1.0"

[package.build]
backend = { name = "pixi-build-rattler-build", version = "*", channels = [
    "https://conda.modular.com/max-nightly",
    "https://repo.prefix.dev/modular-community",
    "https://repo.prefix.dev/conda-forge",
] }

[package.build-dependencies]
max = ">=26.3"

[package.host-dependencies]
max = ">=26.3"
python = ">=3.12"
numpy = ">=2.4"
scipy = ">=1.17"

[package.run-dependencies]
python = ">=3.12"
numpy = ">=2.4"
scipy = ">=1.17"

[dependencies]
# existing deps remain (numpy, scipy, pytest, etc. for dev environment)
fpe-engine = { path = "." }
```

Key points:
- `preview = ["pixi-build"]` is required to opt into the build system
- `fpe-engine = { path = "." }` in `[dependencies]` tells pixi to build the local package for dev
- Existing `[dependencies]` for numpy/scipy/etc. stay for dev environment, but `fpe-engine` is added
- Channels are inside the `backend` dict (pixi v0.68+ convention) with `repo.prefix.dev` URLs
- `max >=26.3` is sufficient (includes mojo compiler); no separate `mojo` pin needed in package deps
- numpy/scipy appear in both `[package.host-dependencies]` and `[package.run-dependencies]` — they're needed at build time for the Python native extension

## Step 2: recipe.yaml

```yaml
context:
  version: "0.1.0"

package:
  name: fpe-engine
  version: ${{ version }}

source:
  - path: .
    use_gitignore: true

build:
  number: 0
  script:
    - export SHLIB_EXT=$(python -c "import sys; print('.dylib' if sys.platform == 'darwin' else '.so')")
    - mkdir -p $PREFIX/lib
    - mojo build -I $SRC_DIR/src --emit shared-lib -o $PREFIX/lib/libfpe_engine$SHLIB_EXT $SRC_DIR/src/bindings/c_abi.mojo
    - |
      if [ "$(uname)" = "Darwin" ]; then
        install_name_tool -id @rpath/libfpe_engine.dylib $PREFIX/lib/libfpe_engine.dylib
      fi
    - mkdir -p $PREFIX/include
    - cp $SRC_DIR/cpp/include/fpe_engine.h $PREFIX/include/
    - |
      PY_VER=$(python -c "import sysconfig; print(sysconfig.get_config_var('py_version_short'))")
      PY_EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
      SITE=$PREFIX/lib/python${PY_VER}/site-packages
      mojo build -I $SRC_DIR/src --emit shared-lib -o _fpe_native${PY_EXT} $SRC_DIR/src/bindings/_fpe_native.mojo
      mkdir -p $SITE/fpe_engine
      cp _fpe_native${PY_EXT} $SITE/fpe_engine/
      cp $SRC_DIR/python/fpe_engine/__init__.py $SITE/fpe_engine/
      cp $SRC_DIR/python/fpe_engine/pricer.py $SITE/fpe_engine/

requirements:
  build:
    - max >=26.3
  host:
    - max >=26.3
    - python >=3.12
    - numpy >=2.4
    - scipy >=1.17
  run:
    - python >=3.12
    - numpy >=2.4
    - scipy >=1.17

test:
  commands:
    - python -c "import fpe_engine; print(fpe_engine.is_available())"
  python:
    imports:
      - fpe_engine

about:
  homepage: https://github.com/knight/fpe-option
  license: MIT
  summary: FPE Option Pricing Engine - Mojo + MAX AI Kernels
  description: |
    Finite Pointset Estimation (FPE) based option pricing engine
    for Heston stochastic volatility model with barrier option support.
    Built with Mojo for high-performance numerical computation.
```

Key points:
- `$SRC_DIR` = source checkout, `$PREFIX` = install prefix (rattler-build provides these)
- `SHLIB_EXT` is explicitly computed from `sys.platform` (not relying on rattler-build to set it)
- Build script replicates `build_cpp.sh` logic using rattler-build env vars
- The C++ example compilation is excluded (not part of the distributable package)
- `use_gitignore: true` prevents shipping `.pixi/`, `build/`, etc. — `.pixi/` and `build/` must be in `.gitignore`

**Gitignore prerequisite:** Verify `.gitignore` contains `.pixi/` and `build/`. If not, add them.

## Step 3: GitHub Actions — CI Workflow

`.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    strategy:
      matrix:
        platform: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.platform }}
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.0
      - run: pixi install
      - run: pixi run build
      - run: pixi run python -m pytest tests/ -v
      - run: pixi run test
```

Notes:
- `pixi run build` runs `build_cpp.sh` (the existing task) for CI validation
- `pixi run test` runs Mojo unit tests
- After migration is complete, consider adding `pixi build` to test the new build system

## Step 4: GitHub Actions — Publish Workflow

`.github/workflows/publish.yml`:

```yaml
name: Publish
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    strategy:
      matrix:
        include:
          - platform: ubuntu-latest
            target: linux-64
          - platform: macos-latest
            target: osx-arm64
    runs-on: ${{ matrix.platform }}
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.0
      - run: pixi publish https://prefix.dev/fpe-engine
```

Key points:
- OIDC trusted publishing — no `PREFIX_DEV_API_KEY` secret needed
- Triggered on version tags (`v0.1.0`, `v0.2.0`, etc.)
- `id-token: write` permission required for OIDC token exchange
- `pixi publish` builds + uploads in one step (positional channel URL, not `--target-channel`)
- Matrix builds both `linux-64` and `osx-arm64` platforms
- First publish: user must create the `fpe-engine` channel on prefix.dev and configure trusted publishing

## Prefix.dev Setup (one-time, manual)

1. Create account at [prefix.dev](https://prefix.dev)
2. Create channel `fpe-engine`
3. Configure trusted publishing:
   - Go to channel Settings → Trusted Publishers → Add
   - Repository: `<owner>/fpe-option` (your GitHub repo)
   - Environment: `publish` (or leave blank for any)
   - Workflow filename: `publish.yml`
4. After first successful publish, packages are available via:
   ```
   conda install -c https://prefix.dev/fpe-engine fpe-engine
   ```

## Follow-ups (out of scope)

- Add `__init__.pyi` type stubs for `pricer.py`'s native module imports
- Consider adding `pixi build` to CI once pixi-build is out of preview
- Add conda-forge staging if wider distribution is needed later

## File Changes Summary

| File | Action |
|---|---|
| `pixi.toml` | Modify — add `[package]`, `preview`, build/host/run deps |
| `recipe.yaml` | Create — rattler-build recipe for fpe-engine |
| `.github/workflows/ci.yml` | Create — PR/push CI |
| `.github/workflows/publish.yml` | Create — tag-triggered publish |
| `.gitignore` | Verify — ensure `.pixi/` and `build/` are listed |
| `scripts/build_cpp.sh` | Keep — local dev fallback |
