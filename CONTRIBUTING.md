# Contributing

## Development Setup

```bash
pixi install
pixi run build
```

## Testing

```bash
pixi run test             # Python binding tests
pixi run test-fpe         # Mojo FPE engine tests
pixi run test-mojo        # Individual Mojo test
```

## Code Style

- Mojo: follow existing patterns, no commented-out code
- Python: PEP 8, type hints for all public functions
- C++: C++17, RAII for resource management

## Pull Request Checklist

- [ ] Tests pass: `pixi run test && pixi run test-fpe`
- [ ] Build succeeds: `pixi run build`
- [ ] No tracked build artifacts (`*.so`, `*.dylib`)
- [ ] Version updated in `pixi.toml` and `recipe.yaml` for releases

## Releasing

1. Update version in `pixi.toml`, `recipe.yaml`, `mojoproject.toml`
2. Update `CHANGELOG.md`
3. `git tag vX.Y.Z && git push origin vX.Y.Z`
4. CI publish workflow pushes to prefix.dev automatically
