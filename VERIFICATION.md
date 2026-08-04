# Verification Record

Verified locally on August 4, 2026.

## Host

- macOS 26.6;
- Apple M3 Max, 40-core GPU;
- ARM64;
- Metal 4;
- Python 3.12.13.

## Result

The Metal preflight passed and all 6 unit tests passed. The suite exercised source ranking, grounded citation output, index serialization, sentiment prediction, held-out accounting, and model persistence. CLI smoke tests also built an index, answered a question with a source citation, trained the sample sentiment model, and wrote both artifacts.

## Reproduce

```bash
./scripts/verify_macos_metal.sh
uv sync --python 3.12
uv run python -m unittest discover -s tests -v
```
