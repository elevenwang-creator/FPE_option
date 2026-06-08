# Changelog

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
