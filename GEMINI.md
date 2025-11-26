# Motor Universal - Lottery Prediction Engine

## Project Overview
**Motor Universal** is a sophisticated data engineering and statistical analysis system designed for auditing and predicting lottery outcomes. Unlike simple random generators, this project employs a rigorous scientific methodology involving multiple phases of analysis:
1.  **Phase 1 (Audit):** Establishes a statistical baseline by analyzing randomness, frequency, and anomalies.
2.  **Structural & Derived:** Transforms raw event data into complex feature sets.
3.  **Fusion:** Integrates multiple predictive models/engines.
4.  **Backtesting:** Validates strategies against historical data.

## Architecture & Tech Stack
*   **Language:** Python 3.x
*   **Orchestration:** [Prefect](https://www.prefect.io/) (Flows & Tasks)
*   **Data Storage:** Parquet (Canonical data), DuckDB (Analytics), PostgreSQL (Metadata), MinIO (Artifacts).
*   **API:** FastAPI (REST endpoints for predictions and diagnostics).
*   **Frontend:** Streamlit (Dashboard for visualization).
*   **Validation:** Great Expectations (Data Quality).
*   **Version Control:** DVC (Data Version Control) & Git.

## Directory Structure
*   `engine/`: Core domain logic (audits, transformations, fusion algorithms).
    *   `audit/`: Randomness tests and statistical validations.
    *   `derived_dynamic/`: Feature engineering.
    *   `fusion/`: Model ensembling logic.
*   `flows/`: Prefect pipelines that orchestrate the `engine` modules.
    *   `daily_pipeline.py`: Main daily execution flow.
*   `apps/streamlit/`: Dashboard application.
*   `docs/`: Comprehensive documentation (ADRs, Runbooks, Audit Reports).
*   `great_expectations/`: Data validation suites.
*   `tests/`: Test suites (Unit & E2E).

## Key Commands & Workflows

### 1. Environment Setup
The project uses a Python virtual environment. Initialize it using the provided PowerShell script:
```powershell
.\env.ps1
```
This script activates the `.venv` and sets up Prefect environment variables (`PREFECT_API_URL`).

### 2. Running the Dashboard
To start the Streamlit dashboard (UI):
```bash
# Using Docker (Recommended)
docker compose --env-file .env -f ops/docker/compose.yml up -d streamlit

# Or locally (requires dependencies)
streamlit run apps/streamlit/app.py
```
Access at: `http://localhost:8501`

### 3. Running Pipelines (Prefect)
The core logic is executed via Prefect flows.
*   **Daily Pipeline:**
    ```python
    python flows/daily_pipeline.py
    ```
    *Or triggered via Prefect deployment.*

### 4. Running Tests
```bash
pytest tests
```

## Development Conventions
*   **Code Style:** Adheres to `Black` and `Ruff` standards.
*   **Data Formats:**
    *   **Dates:** ISO 8601 (`YYYY-MM-DD`).
    *   **Lottery Numbers:** `00` to `99` (Integer). **Never truncate** distributions (always show top-N or full range).
    *   **Positions:** 1, 2, 3 (Standard lottery draw positions).
*   **Logic Separation:** Keep core logic in `engine/` and orchestration/side-effects in `flows/`.
*   **Validation:** All data inputs and outputs should be validated against schemas (Great Expectations).

## Context for AI
*   **Strictness:** This is a financial/statistical tool. Accuracy and reproducibility are paramount.
*   **No Superstition:** Focus on statistical anomalies, bias detection, and pattern recognition, not "luck".
*   **Documentation:** Always check `docs/adr/` before proposing architectural changes.
