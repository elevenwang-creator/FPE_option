# Changelog

## 0.2.1 (2026-06-08)

### Added
- CI: publish workflow triggered by git tags (`v*`) to prefix.dev `fpe-engine` channel
- CI: `max-parallel: 1` to prevent Mojo OOM on Linux runners

### Fixed
- CI: `build_cpp.sh` creates output directory before `mojo build`
- CI: removed manual `pixi install` (handled by `setup-pixi` v0.8.5)
- Recipe: homepage URL corrected to `elevenwang-creator/FPE_option`
- Git history: author email unified to GitHub noreply address
- Test `kron_spmv`: removed nonexistent `kron_T_spmv_dual` import
- Test `new_operators`: removed unused `identity_csr` import
- Test `ddz_z3`: added `.copy()` to avoid mutation side effects
- Test `sparse_lu_symnum`: rewritten to match current `factorize()` API

### Changed
- CI: upcoming-feature tests (calibrator, bindings, e2e, GPU, autograd, etc.) excluded from CI — run locally only
- README: pixi install channel updated to `https://repo.prefix.dev/fpe-engine`

## 0.2.0 (2026-06-08)

### Added
- LICENSE file (MIT)
- CONTRIBUTING.md with development guide

### Fixed
- C++ RAII wrapper: throws `std::runtime_error` on pipeline creation failure instead of silent empty results
- `FpeParams.is_valid()`: up-option barrier boundary check consistent with payoff logic (barrier must be strictly > S0)
- C ABI pipeline: use `heston.S0` as dummy strike instead of hardcoded `100.0`
- GPU NAIS training kernel: raises clear error instead of silently returning wrong results
- `.gitignore`: exclude build artifacts (`*.so`, `*.dylib`, cached binaries)

### Changed
- Version unified to `0.2.0` across all config files
- Cleaned up tracked build artifacts from repository

## 0.1.0 (Unreleased)

- Initial FPE engine implementation
- Mojo native engine with C ABI, C++ RAII wrapper, and Python bindings
- GPU acceleration path for FPE solver
- NAIS neural network engine for FBSDE-based pricing
- Heston model calibration
