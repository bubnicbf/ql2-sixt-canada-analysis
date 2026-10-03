# ql2-sixt-canada-analysis

## Project structure

```text
data/raw/          Immutable source CSVs (proprietary; kept local, not in Git)
data/interim/      Intermediate generated datasets (ignored by Git)
data/processed/    Final analysis-ready datasets (ignored by Git)
notebooks/         Exploratory and presentation notebooks
src/ql2_sixt_canada_analysis/   Reusable Python source package
tests/             Automated validation (pytest)
reports/           Generated analytical reports (ignored by default)
reports/figures/   Generated charts and figures (ignored by default)
docs/              Supporting documentation
```

- `data/raw/` is immutable input data: never edit, rename, or overwrite it.
- Generated data belongs in `data/interim/` or `data/processed/`.
- Reusable logic belongs under `src/ql2_sixt_canada_analysis/`.
- Automated validation belongs under `tests/`.
- Notebooks should import and call functions from the source package rather
  than contain duplicated production logic.

## Environment setup

Requires Python 3.11 or newer. Run these commands from the repository root.

```bash
python3 --version                      # must report 3.11 or newer
python3 -m venv .venv                  # create a local virtual environment
source .venv/bin/activate              # macOS/Linux (Windows: .venv\Scripts\activate)
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"      # package + runtime deps + pytest/JupyterLab
python -m pytest                       # run the test suite
jupyter lab                            # optional: start JupyterLab
```

Dependencies are declared only in `pyproject.toml`: runtime analysis libraries
under `[project] dependencies`, and test and notebook tools under the `dev`
extra. Recreate the environment by deleting `.venv/` and repeating the steps.

Data and environment safety:

- `.venv/` is local to your machine and must never be committed.
- All QL2 data in `data/raw/`, `data/interim/`, and `data/processed/` is
  proprietary and must never be committed; Git ignores these directories
  except for their placeholder/README files.
- Tests use small synthetic data created inside the tests. Never substitute
  real QL2 data for committed test fixtures.
- This configuration installs code and dependencies only; it does not make
  proprietary datasets safe to publish.

## Data trust

- All scheduled cities represented
- All expected locations present in jobs
- Job level counts = detail row counts
- Valid offers duplicated
- Job timestamps consistent
- Date field that defines reporting day
- Location/Product attributes stability

## Airport matched premium

Charge difference between airport and downtown

- City
- Vehicle type
- Capture time
- Price percentile
- Percentage of matches where the airport is higher

## Genuine pricing events

- Matched offer frequency from one hour to the next
- Typical increase/decrease
- Simultaneuous products move count
- Do airport/downtown locations move together
- New price persistence
- Simultaneous assortment change
- Duplicated large moves across aliased locations

## Visible assortment stability

- Offers per location and capture
- Which products appear/disappear
- Isolated/widespread changes
- Consecutive product sets similarity
- Falling assortment vs. price increases

## QL2 controls

- Expected/location completeness
- Job/detail reconciliation
- Duplicate/location/alias detection
- Timestamp offset validation
- Product attribute consistency
- Large synchronized price change alerts
- Assortment drop alerts
- Persistence confirmation