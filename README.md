# Investment Simulator Backend

This backend hosts a Streamlit dashboard for deterministic and Monte Carlo investment simulations.

## Prerequisites

- Python 3.14
- pip

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
streamlit run streamlit_app/main.py
```

## Deploy to Streamlit

Use the production dependency manifest:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app/main.py --server.headless true
```

For hosted Streamlit deployment, set the app entrypoint to:

- `streamlit_app/main.py`

## Quality gates

```bash
ruff check .
ruff format --check .
mypy streamlit_app
pytest
bandit -r streamlit_app
pip-audit
```
