# Notebook workflow

Launch the executable tutorial from this repository root:

```bash
uv sync --locked --group notebook
uv run --group notebook jupyter lab notebooks/01_data_to_validation.ipynb
```

The notebook generates disorganized and cleaned CSVs under `notebooks/data/`, writes runtime models under `notebooks/artifacts/`, and validates the real package plus its unit tests.
