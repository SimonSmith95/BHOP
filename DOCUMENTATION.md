# BHOP — Bayesian Hyperparameter Optimisation for the Lab
## Complete Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation & Setup](#2-installation--setup)
3. [How to Use the Application (GUI Walkthrough)](#3-how-to-use-the-application-gui-walkthrough)
   - 3.1 [Launching the GUI](#31-launching-the-gui)
   - 3.2 [Loading Your CSV File](#32-loading-your-csv-file)
   - 3.3 [Configuring Parameters](#33-configuring-parameters)
   - 3.4 [Setting Objectives](#34-setting-objectives)
   - 3.5 [Choosing Batch Size, n-Batches and Sampler](#35-choosing-batch-size-n-batches-and-sampler)
   - 3.6 [Running the Optimisation Loop](#36-running-the-optimisation-loop)
   - 3.7 [Entering Batch Results](#37-entering-batch-results)
   - 3.8 [Saving and Resuming Sessions](#38-saving-and-resuming-sessions)
   - 3.9 [Design Space Visualisation](#39-design-space-visualisation)
4. [Headless / Scripting Mode](#4-headless--scripting-mode)
5. [Codebase Architecture](#5-codebase-architecture)
   - 5.1 [File Map](#51-file-map)
   - 5.2 [Module Dependency Diagram](#52-module-dependency-diagram)
   - 5.3 [The Ask / Tell Cycle](#53-the-ask--tell-cycle)
6. [Module Reference](#6-module-reference)
   - 6.1 [parameter_config.py](#61-parameter_configpy)
   - 6.2 [csv_loader.py](#62-csv_loaderpy)
   - 6.3 [sampler_utils.py](#63-sampler_utilspy)
   - 6.4 [optuna_builder.py](#64-optuna_builderpy)
   - 6.5 [session_manager.py](#65-session_managerpy)
   - 6.6 [worker.py](#66-workerpy)
   - 6.7 [GUI Widgets](#67-gui-widgets)
   - 6.8 [main.py & BHOP.py](#68-mainpy--bhoppy)
7. [Validation Script (validate_perovskite.py)](#7-validation-script-validate_perovskitepy)
   - 7.1 [What It Validates and Why](#71-what-it-validates-and-why)
   - 7.2 [How to Run It](#72-how-to-run-it)
   - 7.3 [Configuration Constants](#73-configuration-constants)
   - 7.4 [Section-by-Section Breakdown](#74-section-by-section-breakdown)
   - 7.5 [Output Files](#75-output-files)
   - 7.6 [Interpreting the Pass/Fail Report](#76-interpreting-the-passfail-report)
8. [Configuration Reference](#8-configuration-reference)
9. [Common Adjustments & Recipes](#9-common-adjustments--recipes)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Overview

**BHOP** is a lab-facing Bayesian Optimisation tool. It wraps the
[Optuna](https://optuna.org/) optimisation framework in a clean PySide6 GUI,
making it practical for wet-lab or materials-science workflows where:

- Experiments are run in **batches** (you ask the model for N suggestions,
  run them in the lab, then return the measured results).
- The experiment parameters are described by a **CSV file** you already have
  (historical data or an experiment log).
- The user needs to be able to **pause, close, and resume** the optimisation
  across days or weeks.

### What each part does

| File / folder | Role |
|---|---|
| `main.py` | GUI entry point — call `python main.py` |
| `main_window.py` | Full PySide6 application window |
| `parameter_config.py` | Data models shared across the whole app |
| `csv_loader.py` | Reads CSV, auto-detects column types |
| `sampler_utils.py` | Dead-region-aware parameter sampling |
| `optuna_builder.py` | Creates the Optuna study, ask/tell API |
| `session_manager.py` | Saves / loads session state to JSON + SQLite |
| `worker.py` | Background QThread running the BO loop |
| `param_card_widget.py` | UI widget for one parameter |
| `batch_results_dialog.py` | Dialog to enter lab measurements |
| `dead_region_dialog.py` | Dialog to define forbidden parameter zones |
| `BHOP.py` | Headless scripting example (no GUI needed) |
| `test_backend.py` | Pytest test suite for all backend modules |
| `validate_perovskite.py` | Stand-alone validation suite for the perovskite dataset |
| `test_data/` | Example CSV and auto-generated session files |
| `validation_results/` | PNG plots and metrics CSV from the validation script |

---

## 2. Installation & Setup

### Requirements

- **Python 3.10+** (tested on 3.14)
- All dependencies in `requirements.txt`

### Install

```bash
pip install -r requirements.txt
```

The dependencies are:

| Package | Version | Purpose |
|---|---|---|
| `optuna` | ≥ 3.6 | Bayesian optimisation engine |
| `pyside6` | ≥ 6.7 | GUI framework |
| `pandas` | ≥ 2.0 | CSV handling |
| `numpy` | ≥ 1.26 | Numerical operations |
| `matplotlib` | ≥ 3.8 | Plots (validation script) |
| `sqlalchemy` | ≥ 2.0 | Optuna's SQLite backend |
| `scikit-learn` | ≥ 1.4 | RF + GP surrogates (validation script) |

> **Note for Windows users:** If you see a `UnicodeEncodeError` when running
> scripts in the terminal, the script already adds `sys.stdout.reconfigure`
> at the top to handle this. See [Troubleshooting](#10-troubleshooting).

---

## 3. How to Use the Application (GUI Walkthrough)

### 3.1 Launching the GUI

```bash
python main.py
```

The main window opens with an empty state. You will see:
- **Left dock — ⚙ Settings tab** — Objectives selector, sampler, batch settings, warnings, Ask / Pause buttons
- **Left dock — 📊 Results tab** — All Trials table, Best/Pareto table, Design Space button, Export button (auto-activated after each batch)
- **Centre panel** — scrollable parameter cards (one per input column in your CSV)
- **Toolbar** — Load CSV, Reload CSV, 📋 Enter Pending Results…
- **Status bar** — trial count, best value, current status

---

### 3.2 Loading Your CSV File

Go to **File → Load CSV** (or click the toolbar icon).

Your CSV must have:
- **One column per experiment parameter** (inputs).
- **One or more result columns** (outputs / objectives).
- Optionally: rows where result columns are blank — those are treated as
  pending experiments and can be used to warm-start the model.

**Example format:**

```csv
spin_speed,concentration,hotplate_temp,thickness_nm
2000,0.05,80,125.3
3000,0.08,100,89.7
1500,0.03,60,210.5
```

After loading, the app:
1. Infers the type of each column (`INT`, `FLOAT`, `CATEGORICAL`, `BOOL`).
2. Creates a **parameter card** in the left dock for each non-result column.
3. Displays the result columns in the **Objectives** selector at the top.

> **Type inference rules** (from `csv_loader.infer_column_type`):
> - Bool if only {0, 1} or {True, False} values.
> - Categorical if object/string dtype or ≤ 8 unique numeric values.
> - INT if all values equal their integer cast.
> - FLOAT otherwise.

---

### 3.3 Configuring Parameters

Each parameter column gets a **parameter card** in the left dock. You can:

| Control | What it does |
|---|---|
| **Enable toggle** | Uncheck to hold this parameter fixed (excluded from optimisation) |
| **Type selector** | Override the auto-detected type if needed |
| **Min / Max fields** | The full range Optuna is allowed to suggest within |
| **Dead Regions button** | Open a dialog to mark forbidden sub-ranges |
| **Categorical choices** | Tick/untick which categories are allowed |
| **Bool toggle** | Set True, False, or "Optimize" (let the model choose) |

#### Dead Regions

Click **Dead Regions** on any numeric parameter to open the Dead Region
Dialog. Here you can define intervals (e.g. `[2800, 3200]` for a spin coater
speed that is mechanically unsafe). The sampler will **never** suggest values
inside a dead region.

Internally, dead regions are stored as the *complement* — a list of
`AllowedSubRange` objects that cover everything *outside* the dead zones.

---

### 3.4 Setting Objectives

At the top of the window, use the **Objectives** selector to choose:
- **Which column** is the result you want to optimise.
- **Direction**: `minimize` (e.g. defect density) or `maximize`
  (e.g. device efficiency).

For **multi-objective** optimisation, select more than one result column.
The sampler switches automatically to NSGA-II and a Pareto-front table
replaces the single-best display.

---

### 3.5 Choosing Batch Size, n-Batches and Sampler

| Setting | Description |
|---|---|
| **Batch size** | How many experiments to suggest per round |
| **Number of batches** | Total rounds of ask → experiment → tell |
| **Sampler** | `TPE` (recommended), `NSGAII` (multi-obj), `Random` (baseline) |

> **TPE** (Tree-structured Parzen Estimator) is the default and works well
> for most single-objective problems. It learns a probabilistic model of the
> objective and balances exploration vs. exploitation automatically.

---

### 3.6 Running the Optimisation Loop

Click **Ask Next Batch** in the toolbar (or press the shortcut).

The app:
1. Starts the `OptimizationWorker` thread.
2. The worker calls `ask_batch()` — Optuna suggests `batch_size` parameter
   sets based on all previous results.
3. A **Batch Results Dialog** opens, showing the suggested parameter values.
4. The suggested parameters are also written to `pending_batch.csv` in the
   session folder — you can hand this to the lab as a work order.

---

### 3.7 Entering Batch Results

After running your experiments, come back to the dialog and:

#### Correcting actual compositions (important for real experiments)

The parameter columns in the dialog show the **suggested target values** with an
amber background. In practice, lab equipment cannot always hit a target with
100 % accuracy. **You can double-click any parameter cell to edit it** and enter
the composition you actually achieved. This ensures Optuna learns from what
was *really* measured, not the idealized target.

> **Rule of thumb:** if your composition error is < ~5 % relative (e.g. you
> targeted 0.15 and achieved 0.14–0.16), it is fine to leave the values
> unchanged. Larger errors (> 5–10 %) should be corrected before submitting
> so the surrogate model trains on accurate data.

#### Entering measurement results

- **Type the measured values** directly into the objective (white) cells, or
- Click **⬇ Import Results from CSV** to load a CSV that already has the
  results filled in. The app matches rows by parameter values and auto-fills
  the cells.

Click **Submit All** to tell Optuna the results. The app then:
1. Appends the actual compositions + results to your experiment CSV (so the
   CSV always reflects the full history, including any composition corrections).
2. Calls `tell_batch()` to store the results in the Optuna SQLite DB.
3. Moves to the next batch (ask → experiment → tell).
4. Updates the trials table in the centre panel.

Click **Cancel** to close the dialog without submitting. The pending batch is
still saved — see §3.8 for how to come back to it later.

---

### 3.8 Saving and Resuming Sessions

BHOP **automatically saves** the session after every ask and every tell.
The session is stored as two files in the same folder as your CSV (or a chosen
directory):

```
bhop_study_20260822_232040.db           ← Optuna SQLite database
bhop_study_20260822_232040_session.json ← Session metadata + pending batch
```

To **resume**:
- Go to **File → Load Session** and pick the `_session.json` file, or
- Use **File → Recent Sessions** to open a previously used session.

The app reloads the config, reconnects to the SQLite DB, and if there was a
pending batch it shows a banner at the top of the screen.

#### Entering results for a pending batch after coming back from the lab

**Option 1 — Toolbar button (recommended):** Click **📋 Enter Pending Results…**
in the toolbar (it is enabled whenever a pending batch exists). The batch
results dialog opens directly. The parameter cells show the original suggested
target values in amber — **double-click any cell to edit it** if the lab
achieved a different composition. Fill in your measurements and click Submit.

**Option 2 — CSV auto-fill:** Click **Load Updated CSV…** in the resume banner.
Select a CSV file that already has the measured results in the objective column(s).
The app matches rows by parameter values and auto-fills the dialog.

> **Important:** Always keep the `.db` and `_session.json` files together in
> the same folder. Moving one without the other will cause a
> `FileNotFoundError` on load.

> **Never start a new session** to submit results for a pending batch. Always
> reload the original session — this preserves the full Optuna learning history.

---

### 3.9 Design Space Visualisation

The **Design Space** window lets you inspect where the historical data
lives in parameter space and — as soon as a batch is suggested — see
whether the suggested experiments sit sensibly within that space or are
dangerously far from any observed data.

#### Opening the window

Switch to the **📊 Results** tab in the left dock, then click **📊 Design
Space…** (top-left of the Results tab). The window opens as a separate,
resizable, non-modal dialog — you can keep it open alongside BHOP and switch
between them.

The window remembers its size between uses within the same session and is
capped to fit your screen on first open.

#### Two tabs inside the dialog

The Design Space dialog has two tabs:

| Tab | Contents |
|---|---|
| **📊 Design Space** | Pairplot / parallel coordinates / marginals — where the data lives in parameter space |
| **🔗 Correlation Matrix** | Annotated heatmap showing Pearson or Spearman correlations between all numeric parameters and all objectives |

Switch between them freely while the dialog is open.

#### Correlation Matrix tab

The correlation matrix shows **which input parameters drive each objective**
and how parameters correlate with each other:

- Colour scale: `RdBu_r` — blue = strong negative (–1), white = none (0), red = strong positive (+1)
- Each cell is annotated with the numeric correlation value.
- A dashed line separates the **parameter block** (rows/columns) from the
  **objective block**, making it easy to focus on the input → output relationships.
- Use the **Pearson / Spearman** dropdown to switch methods:
  - **Pearson** — linear correlation (fast, standard, sensitive to outliers).
  - **Spearman** — rank-based (robust to outliers, catches monotonic non-linear trends).

**What to look for:**
- A cell near **±1** between a parameter and the objective → that parameter has a strong effect.
- A cell near **0** → that parameter is relatively unimportant for the objective.
- High correlations *between parameters* can indicate redundant inputs or constraint boundaries.

#### What is plotted (Design Space tab)

The plot type is chosen automatically based on the number of enabled
numeric (`INT` or `FLOAT`) parameters in your study:

| # enabled numeric params | Plot type |
|---|---|
| ≤ 5 | **Pairwise scatter grid** — every pair of parameters as a lower-triangle scatter. Diagonal cells show histograms. |
| 6–12 | **Parallel coordinates** — one vertical axis per parameter; historical trials as translucent coloured lines. |
| > 12 | **1-D marginal strip** — one histogram per parameter stacked vertically. |

All historical data points are **coloured by objective value** on a
`RdYlGn` colourmap:
- **Minimize** direction → green = low (good), red = high (bad).
- **Maximize** direction → reversed (green = high = good).

#### Suggested batch overlay

When you click "Ask Next Batch" and the batch is ready, the design space
is automatically updated to overlay the suggestions **before** the results
dialog opens. Suggested points appear as:

- **Pairplot** — pink ★ stars with a white outline at each pair-coordinate.
- **Parallel coords** — thick bright lines (one colour per suggestion).
- **Marginals** — dashed vertical lines at the suggested value.

This lets you immediately judge:
- **Are the suggestions inside the data cloud?** — If stars/lines sit far
  outside the historical data cloud, the model is extrapolating. Be cautious.
- **Are they heading towards good (green) regions?** — If suggestions cluster
  near green points, the optimizer is exploiting known good areas.
- **Are they spread out?** — Spread-out suggestions indicate the optimizer is
  exploring; clustered suggestions indicate convergence/exploitation.

#### Toolbar controls

| Control | Effect |
|---|---|
| **Colour by objective** dropdown | Switch which objective drives the colour scale (relevant when multiple objectives are defined). |
| **⟳ Refresh** button | Manually force a redraw (e.g. after changing a parameter's enabled/disabled state). |

#### When does it auto-refresh?

| Event | Design Space behaviour |
|---|---|
| CSV loaded + objectives applied | Redraws with historical data, no suggestions |
| "Ask Next Batch" completes | Suggestions overlaid in pink |
| Window opened via button | Always redraws with the latest data |

The window is **not** automatically updated while it is closed. Opening it
again after new batches are submitted will show the current state.

---

## 4. Headless / Scripting Mode

`BHOP.py` is a self-contained example showing how to use the backend without
any GUI. This is useful for automated pipelines or running from a script.

```python
# Minimal pattern (see BHOP.py for the full annotated example)
from parameter_config import StudyConfig, ParameterConfig, ParameterType, ObjectiveConfig
from optuna_builder import build_study, ask_batch, tell_batch, load_historical_trials
from session_manager import SessionManager

# 1. Define your search space
config = StudyConfig(
    parameters=[
        ParameterConfig(name="x", ptype=ParameterType.FLOAT, full_min=0.0, full_max=1.0),
    ],
    objectives=[ObjectiveConfig(column_name="y", direction="minimize")],
    batch_size=3,
    n_batches=5,
    sampler_name="TPE",
)

# 2. Create a session
state = SessionManager.create_new_session(config, csv_path="", session_dir="/tmp/my_run")
study = build_study(config, state.storage_path, state.study_name)

# 3. Seed with historical data (optional)
load_historical_trials(study, [{"params": {"x": 0.5}, "values": [1.23]}], config)

# 4. Ask / measure / tell loop
for _ in range(config.n_batches):
    trials = ask_batch(study, config, config.batch_size)
    results = [{"trial_number": t.number, "values": [my_measure(t.params)]}
               for t in trials]
    tell_batch(study, [r["trial_number"] for r in results],
               [r["values"] for r in results])

print("Best:", study.best_trial.params, "→", study.best_trial.value)
```

Replace `my_measure(params)` with your actual experiment or simulation.

---

## 5. Codebase Architecture

### 5.1 File Map

```
PythonProject1/
│
├── main.py                  Entry point (GUI)
├── main_window.py           MainWindow (QMainWindow) — all GUI logic
│
├── parameter_config.py      Data models — shared by every other module
├── csv_loader.py            CSV I/O + type inference
├── sampler_utils.py         Dead-region-aware Optuna suggestion helpers
├── optuna_builder.py        Study creation + ask/tell API
├── session_manager.py       Save/load JSON session + SQLite path tracking
├── worker.py                QThread running the batch loop
│
├── param_card_widget.py     UI: one card per parameter in the left dock
├── batch_results_dialog.py  UI: dialog for entering batch results
├── dead_region_dialog.py    UI: dialog for defining dead regions
│
├── BHOP.py                  Headless scripting example
├── test_backend.py          Pytest test suite
├── validate_perovskite.py   Stand-alone validation suite (perovskite data)
├── requirements.txt         pip dependencies
│
└── test_data/
    ├── Perovskite_dataset.csv     Example perovskite stability dataset
    ├── *.db                       Auto-generated Optuna SQLite databases
    ├── *_session.json             Auto-generated session files
    └── pending_batch.csv          Auto-generated lab work order
```

---

### 5.2 Module Dependency Diagram

```
                    ┌─────────────────┐
                    │  main.py        │
                    │  (entry point)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  main_window.py │◄──── param_card_widget.py
                    │  (GUI)          │◄──── batch_results_dialog.py
                    └──┬──────────┬──┘◄──── dead_region_dialog.py
                       │          │
           ┌───────────▼──┐  ┌────▼──────────────┐
           │  worker.py   │  │  session_manager.py│
           │  (QThread)   │  │                    │
           └───────┬──────┘  └────────────────────┘
                   │                    │
          ┌────────▼──────────┐         │
          │  optuna_builder.py│◄────────┘
          │  (ask / tell)     │
          └──────┬────────────┘
                 │
      ┌──────────▼──────────────────────────────┐
      │            parameter_config.py           │
      │  (ParameterConfig, StudyConfig, etc.)    │
      └──────────┬───────────────────────────────┘
                 │                │
     ┌───────────▼──┐    ┌────────▼───────┐
     │ csv_loader.py│    │sampler_utils.py│
     │              │    │                │
     └──────────────┘    └────────────────┘
```

All backend modules (`optuna_builder`, `session_manager`, `csv_loader`,
`sampler_utils`) import only from `parameter_config` and standard library /
third-party packages — they have **no dependency on PySide6** and can be used
without a GUI.

---

### 5.3 The Ask / Tell Cycle

One "round" of optimisation looks like this:

```
 Worker thread                 Main thread (GUI)             User / Lab
 ─────────────                 ─────────────────             ──────────
 ask_batch()
   Optuna suggests N params
   → N Trial objects
 mark_batch_pending()
   Saves session JSON
   Writes pending_batch.csv ──────────────────────────────► Lab runs experiments
 emit batch_ready signal
                               BatchResultsDialog opens
                               (auto-fills from CSV if available)
                                                             User enters values
                               submit_results() called
 [unblocked by Event]
 tell_batch()
   Results stored in SQLite
 clear_pending_batch()
   Removes pending_batch.csv
 emit batch_complete signal
                               UI tables updated
```

This cycle repeats `n_batches` times, then `optimization_done` is emitted.

---

## 6. Module Reference

### 6.1 `parameter_config.py`

Defines all shared data models. **No external dependencies** beyond the
standard library.

#### `ParameterType` (Enum)

```python
class ParameterType(Enum):
    INT         = "INT"
    FLOAT       = "FLOAT"
    BOOL        = "BOOL"
    CATEGORICAL = "CATEGORICAL"
```

#### `AllowedSubRange`

A single continuous allowed interval `[low, high]`.

```python
@dataclass
class AllowedSubRange:
    low: float
    high: float
```

Used in `ParameterConfig.allowed_subranges` to describe the valid parameter
space after dead regions have been removed.

#### `ParameterConfig`

Full configuration for one experiment parameter.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Column name in the CSV |
| `ptype` | `ParameterType` | Data type |
| `enabled` | `bool` | If False, excluded from optimisation |
| `full_min` | `float` | Absolute minimum (INT / FLOAT) |
| `full_max` | `float` | Absolute maximum (INT / FLOAT) |
| `allowed_subranges` | `List[AllowedSubRange]` | Valid intervals (complement of dead zones). Defaults to `[full_min, full_max]` |
| `step` | `Optional[int]` | Step size for INT params (None = 1) |
| `all_choices` | `List[str]` | All unique values seen in CSV (CATEGORICAL / BOOL) |
| `allowed_choices` | `List[str]` | Subset the user allows (CATEGORICAL) |
| `fixed_value` | `Optional[bool]` | For BOOL: None = optimise, True/False = hold fixed |

Serialisation: `to_dict()` / `from_dict()` for JSON persistence.

#### `ObjectiveConfig`

```python
@dataclass
class ObjectiveConfig:
    column_name: str
    direction: Literal["minimize", "maximize"] = "minimize"
```

#### `StudyConfig`

Top-level configuration for an entire optimisation session.

| Field | Type | Default | Description |
|---|---|---|---|
| `parameters` | `List[ParameterConfig]` | `[]` | All input parameters |
| `objectives` | `List[ObjectiveConfig]` | `[]` | Result columns to optimise |
| `batch_size` | `int` | `1` | Suggestions per round |
| `n_batches` | `int` | `10` | Total rounds |
| `sampler_name` | `str` | `"TPE"` | `"TPE"`, `"NSGAII"`, or `"Random"` |

---

### 6.2 `csv_loader.py`

Handles all CSV I/O and auto-detection logic.

#### `load_csv(path) → pd.DataFrame`

Reads a CSV and strips whitespace from column names.

#### `infer_column_type(series, categorical_threshold=8) → ParameterType`

Auto-detects the parameter type of a pandas Series using this priority order:

1. **BOOL** — dtype bool, or ≤ 2 unique values all in `{0, 1, True, False}`
2. **CATEGORICAL** — object/string dtype
3. **CATEGORICAL** — numeric but ≤ `categorical_threshold` unique values
4. **INT** — all non-null values equal their integer cast
5. **FLOAT** — everything else

**To change the threshold** for when a numeric column is treated as
categorical, pass `categorical_threshold=N` when calling
`extract_param_defaults()`.

#### `extract_param_defaults(df, result_columns, categorical_threshold=8) → List[ParameterConfig]`

Builds a `ParameterConfig` for each non-result column using the inferred
types and the min/max (or unique choices) found in the data.

#### `load_trials_from_csv(df, config) → List[dict]`

Converts complete rows (where all objective columns have values) into a list
of `{"params": {...}, "values": [...]}` dicts for seeding into Optuna.

#### `append_rows_to_csv(path, rows)`

Appends new result rows to the experiment CSV. Called after each batch is
submitted, so the CSV always reflects the full history.

---

### 6.3 `sampler_utils.py`

Dead-region-aware helpers for Optuna parameter suggestion.

#### The Dead-Region Problem

Standard Optuna `suggest_int(name, low, high)` cannot skip intervals. For
parameters with forbidden zones (e.g. a spin coater that vibrates dangerously
between 2800–3200 rpm), the solution is a **two-level suggestion**:

1. **Level 1 (categorical):** pick which sub-range to sample from, weighted
   by the number of valid values each sub-range contains.
2. **Level 2 (int/float):** sample the actual value within the chosen
   sub-range.

This approach is compatible with Optuna's TPE Parzen estimator.

#### `subrange_suggest_int(trial, name, subranges, step=1) → int`

Suggests an integer from the union of `subranges`. With a single sub-range
this is identical to `trial.suggest_int()`.

#### `subrange_suggest_float(trial, name, subranges) → float`

Same but for floats, weighted by interval width.

#### `compute_allowed_subranges(full_min, full_max, dead_regions) → List[AllowedSubRange]`

Computes the complement of `dead_regions` within `[full_min, full_max]`.

```python
# Example: spin speed 1000–5000, dead zone 2800–3200
allowed = compute_allowed_subranges(1000, 5000, [(2800, 3200)])
# → [AllowedSubRange(1000, 2800), AllowedSubRange(3200, 5000)]
```

Dead regions are **clipped, sorted, and merged** before inversion, so
overlapping or out-of-bounds inputs are handled gracefully.

Raises `ValueError` if dead regions cover the entire range.

#### `validate_subranges(subranges, full_min, full_max) → List[str]`

Returns a list of human-readable error strings (empty = valid). Checks for:
- Empty list
- Zero-width sub-ranges (`low >= high`)
- Out-of-bounds sub-ranges
- Overlapping sub-ranges

---

### 6.4 `optuna_builder.py`

Translates `StudyConfig` into a live Optuna study and provides the
ask/tell batch interface.

#### `build_study(config, storage_path, study_name) → optuna.Study`

Creates (or reloads) a persistent Optuna study backed by a SQLite database.

- **`storage_path`** — absolute path to the `.db` file.
- **`study_name`** — unique name; `load_if_exists=True` means this is safe to
  call on every app start.
- Sampler is chosen by `config.sampler_name`:
  - `"TPE"` → `TPESampler(seed=42, multivariate=True)`
  - `"NSGAII"` → `NSGAIISampler(seed=42)` (multi-objective only)
  - `"Random"` → `RandomSampler(seed=42)`

#### `build_distributions(config) → Dict[str, BaseDistribution]`

Returns the Optuna distributions dict that matches the internal parameter key
space. Required by `study.ask()` and `load_historical_trials()`.

For multi-sub-range parameters the keys are:
- `"{name}__range_idx"` — `CategoricalDistribution` (which sub-range)
- `"{name}__in_range_{idx}"` — `IntDistribution` or `FloatDistribution`

#### `ask_batch(study, config, batch_size) → List[optuna.Trial]`

Asks Optuna for `batch_size` suggestions without running any objective.
Returns `Trial` objects, each with `.number` and `.params`.

#### `tell_batch(study, trial_numbers, results)`

Reports measured results back for the given trial numbers.

- `trial_numbers`: list of `int`
- `results`: list of `[float]` (single objective) or `[float, float, ...]`
  (multi-objective)

#### `load_historical_trials(study, trial_dicts, config) → (added, skipped)`

Adds completed trials from a list of `{"params": {...}, "values": [...]}` dicts
to the study. Deduplicates by parameter fingerprint so it is safe to call
after every CSV reload.

Returns `(number_added, number_skipped)`.

#### `get_pareto_front(study) → List[FrozenTrial]`

For multi-objective studies returns the Pareto front; for single-objective
returns `[study.best_trial]`.

---

### 6.5 `session_manager.py`

Provides save/load persistence for the entire application state.

#### `SessionState` dataclass

| Field | Type | Description |
|---|---|---|
| `session_path` | `str` | Absolute path to the `_session.json` file |
| `storage_path` | `str` | Absolute path to the Optuna `.db` file |
| `study_name` | `str` | Unique Optuna study name |
| `csv_path` | `str` | Path to the experiment data CSV |
| `study_config` | `StudyConfig` | Full configuration |
| `pending_batch` | `Optional[List[dict]]` | Trials asked but not yet told |

`pending_batch` entries have the shape:
```python
{"trial_number": int, "params": {"param_name": value, ...}}
```

#### `SessionManager` (static methods)

| Method | Description |
|---|---|
| `create_new_session(config, csv_path, session_dir)` | Creates a new session and returns the initial `SessionState`. The SQLite DB is not created yet — Optuna creates it lazily. |
| `save(state)` | Writes state to disk **atomically** (write-to-.tmp then rename) to prevent corruption. |
| `load(session_path)` | Loads a `_session.json`. Raises `FileNotFoundError` if the paired `.db` is missing. |
| `list_recent_sessions(n=5)` | Returns the N most recent sessions from `~/.bhop_sessions.json`, filtering out any that no longer exist on disk. |
| `mark_batch_pending(state, trials)` | Saves the pending batch to the session JSON and writes `pending_batch.csv` for the lab. |
| `clear_pending_batch(state)` | Clears the pending batch and deletes `pending_batch.csv`. |
| `match_pending_to_csv(state, df)` | Tries to match pending trial parameters against rows in a DataFrame (for auto-filling result dialogs). Returns `{trial_number: [values]}` or `None`. |

The global recent-session registry is stored at `~/.bhop_sessions.json`
(user's home directory).

---

### 6.6 `worker.py`

A `QThread` subclass that runs the batch optimisation loop in the background,
keeping the GUI responsive.

#### Signals

| Signal | Payload | When emitted |
|---|---|---|
| `batch_ready` | `List[dict]` — `{"trial_number": int, "params": dict}` | After `ask_batch()` completes; GUI should open `BatchResultsDialog` |
| `batch_complete` | `(batches_done: int, total_batches: int)` | After `tell_batch()` completes |
| `optimization_done` | — | All batches done or `stop()` was called |
| `error` | `str` — error message | On unhandled exception |

#### Thread-safety pattern

The worker waits on a `threading.Event` (`_results_ready`) after emitting
`batch_ready`. The main thread calls `submit_results()` or `cancel_results()`
from the results dialog, which sets the event and unblocks the worker.

#### Control methods (call from the main thread)

| Method | Effect |
|---|---|
| `pause()` | Blocks the loop at the next pause-check (before each ask) |
| `resume()` | Unblocks the loop |
| `stop()` | Sets the stop flag; also unblocks any waiting events so the thread exits cleanly |
| `submit_results(results)` | Delivers results and unblocks the worker |
| `cancel_results()` | Cancels the current batch and stops the loop |

---

### 6.7 GUI Widgets

#### `main_window.py — MainWindow`

The top-level `QMainWindow`. Key responsibilities:

- **`_do_load_csv(path)`** — loads CSV, infers column types, builds parameter
  cards, auto-loads historical trials into a new or existing study.
- **`_action_ask_next_batch()`** — syncs config from UI, creates/resumes
  study, starts `OptimizationWorker`.
- **`_on_batch_ready(param_dicts)`** — opens `BatchResultsDialog`; tries to
  auto-fill from the CSV via `SessionManager.match_pending_to_csv()`.
- **`_on_results_submitted(results)`** — calls `worker.submit_results()`.
- **`_on_batch_complete(done, total)`** — refreshes results tables and
  auto-switches the left dock to the **📊 Results** tab so the user sees
  the new data without any manual clicking.
- **`_refresh_results_tables()`** — updates the two `QTableWidget`s in the
  **📊 Results** tab of the left dock (All Trials + Best/Pareto).

Left dock layout (two-tab `QTabWidget`):

| Tab | Contents |
|---|---|
| **⚙ Settings** | Objectives group, Sampler group, Batch Settings group, warning label, Ask Next Batch button, Pause button |
| **📊 Results** | Design Space button, Export CSV button, All Trials table, Best/Pareto table |

#### `design_space_widget.py — DesignSpaceWidget / CorrelationWidget / DesignSpaceDialog`

**`DesignSpaceWidget`** — embeds a Matplotlib figure showing historical data
in parameter space. Plot type is chosen automatically:

| # numeric params | Plot type |
|---|---|
| ≤ 5 | Pairwise lower-triangle scatter grid with diagonal histograms |
| 6–12 | Parallel coordinates (one vertical axis per parameter) |
| > 12 | 1-D marginal strip (stacked histograms) |

Key methods:
- **`refresh(df, params, objectives, suggestions=None)`** — redraws the plot.
- **`clear()`** — hides the canvas and shows the placeholder.
- **`_add_colorbar(sm, ax_or_axes, label)`** — attaches a colorbar with
  human-readable tick labels (e.g. "1.4M" instead of "1.4 × 10⁶") so the
  axis label never overlaps the offset text.

**`CorrelationWidget`** — embeds a Matplotlib heatmap showing pairwise
Pearson or Spearman correlations between all numeric parameters and all
objective columns.

Key features:
- **Pearson / Spearman** dropdown — switch method without reopening.
- **⟳ Refresh** button — manual redraw.
- Cells annotated with the numeric correlation value (bold).
- Dashed line separates the parameter block from the objective block.
- Colours scale from blue (–1) → white (0) → red (+1) using `RdBu_r`.

Key methods:
- **`refresh(df, params, objectives)`** — recomputes and redraws.
- **`clear()`** — resets the canvas.

**`DesignSpaceDialog`** — non-modal `QDialog` wrapping both widgets in a
`QTabWidget` ("📊 Design Space" | "🔗 Correlation Matrix"). Capped to the
screen size on first open; freely resizable thereafter.

Key methods:
- **`refresh(df, params, objectives, suggestions=None)`** — calls both
  `DesignSpaceWidget.refresh()` and `CorrelationWidget.refresh()`.
- **`clear()`** — calls both widgets' `clear()`.
- **`hide()`** — hides the window without destroying data (reopen is instant).

#### `param_card_widget.py — ParamCardWidget`

A `QFrame` representing one parameter. It:
- Shows different controls depending on `ParameterType`
  (spin boxes for numeric, checkboxes for categorical, radio buttons for bool).
- Has a **Dead Regions** button that opens `DeadRegionDialog`.
- Exposes `get_config() → ParameterConfig` to read the current UI state.
- Exposes `set_config(ParameterConfig)` to populate from a saved session.

#### `batch_results_dialog.py — BatchResultsDialog`

A `QDialog` showing a table of suggested parameters (read-only) and editable
result cells. Users can type values directly or import from a CSV.
Emits the filled-in results list when **Submit** is clicked.

#### `dead_region_dialog.py — DeadRegionDialog`

A `QDialog` for defining forbidden intervals on a numeric parameter.
- Displays existing sub-ranges and allows adding/removing rows.
- Shows a live **preview** of the allowed ranges as the user edits.
- Validates with `sampler_utils.validate_subranges()` before accepting.
- For categorical parameters, shows a checklist to un-allow specific values.

---

### 6.8 `main.py` & `BHOP.py`

#### `main.py`

```python
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

Simple entry point. Run with `python main.py`.

#### `BHOP.py`

Contains `run_headless_example()` — a fully annotated scripting template.
Demonstrates:
1. Building a `StudyConfig` with dead regions.
2. Creating a session in a temp directory.
3. Seeding 3 historical data points.
4. Asking for a batch.
5. Measuring with a mock function.
6. Telling the results.

Run with `python BHOP.py` to see it in action without the GUI.

---

## 7. Validation Script (`validate_perovskite.py`)

This script verifies that the surrogate model and BO suggestions are reliable
**before** using the optimiser in a real experiment campaign, using the fully
measured perovskite stability dataset as ground truth.

### 7.1 What It Validates and Why

Because the dataset (`Perovskite_dataset.csv`) is closed and fully measured,
it can be treated as ground truth. Five rigorous checks are run:

| Check | Why it matters |
|---|---|
| **§0 Sanity checks** | Confirm the data is internally consistent (composition sums, replicates) |
| **§1 Surrogate hold-out** | Verify the model can *predict* held-out compositions — if it can't, its suggestions are worthless |
| **§2 Virtual BO** | Simulate the full BO loop on the dataset and compare against random sampling — the gold-standard benchmark |
| **§3 Acquisition surface** | Check that suggestions are physically valid and the model landscape looks plausible |
| **§4 Domain knowledge** | Cross-check predictions against known perovskite chemistry (MA-rich = unstable) |

---

### 7.2 How to Run It

From the project root directory:

```bash
# Windows (adjust path to your python.exe if needed)
"C:\Users\simon\AppData\Local\Python\pythoncore-3.14-64\python.exe" validate_perovskite.py

# macOS / Linux
python validate_perovskite.py
```

Runtime is approximately **3–8 minutes** depending on CPU speed (the repeated
5-fold CV and 10 × 30-step virtual BO are the slow parts).

All plots are saved to `validation_results/` and a metrics table is saved to
`validation_results/surrogate_metrics.csv`. The console prints a pass/fail
checklist at the end.

---

### 7.3 Configuration Constants

All tunable knobs are at the top of the file under `# CONFIGURATION`:

```python
CSV_PATH   = "test_data/Perovskite_dataset.csv"  # path to your dataset
OUT_DIR    = "validation_results"                 # output folder for plots

SEED       = 42       # random seed for reproducibility
TEST_SIZE  = 0.30     # fraction held out for the 70/30 split
N_CV_FOLDS = 5        # k in k-fold cross-validation
N_CV_REPS  = 5        # how many times to repeat the k-fold CV

N_SEED_BO  = 8        # initial random points before BO starts
N_BO_ITER  = 30       # BO steps per run (after seeding)
N_BO_RUNS  = 10       # independent BO runs for confidence bands
GRID_STEP  = 0.025    # resolution of the simplex grid (smaller = finer heatmap)
EI_XI      = 0.01     # Expected Improvement exploration bonus
                      # (larger = more exploration, smaller = more exploitation)

FEATURES   = ["CsPbI", "FAPbI", "MAPbI"]   # input column names
TARGET     = "Instability index"             # result column name
```

**Common adjustments:**

| What to change | Which constant | Effect |
|---|---|---|
| Use a different dataset | `CSV_PATH`, `FEATURES`, `TARGET` | Points at a different CSV |
| Faster run (less accurate) | Reduce `N_CV_REPS`, `N_BO_RUNS` | Fewer repeat runs |
| Finer heatmap | Reduce `GRID_STEP` (e.g. `0.01`) | More grid points, slower |
| More exploration in BO | Increase `EI_XI` (e.g. `0.1`) | Wider uncertainty bonus |
| Stricter surrogate test | Increase `TEST_SIZE` | Fewer training points = harder test |

---

### 7.4 Section-by-Section Breakdown

#### §0 — Data Loading & Sanity Checks (`section0`)

1. Loads the CSV and prints basic stats (row count, min/max/mean of target).
2. **Composition sum check**: verifies `CsPbI + FAPbI + MAPbI ≈ 1.0` for
   every row (tolerance ±0.02). Flags any violations.
3. **Replicate analysis**: groups rows by composition, counts replicates,
   reports max within-group standard deviation and median coefficient of
   variation.
4. Shows the top-5 compositions by mean instability index.
5. **Saves two plots:**
   - `s0_ternary_raw.png` — ternary scatter of all measured compositions,
     coloured by mean instability (log scale, green = stable).
   - `s0_replicate_variance.png` — same scatter but coloured by within-group
     standard deviation (shows where noise is highest).

#### §1 — Surrogate Hold-out Validation (`section1`)

1. Averages replicates → one value per unique composition.
2. **(A) 70/30 hold-out split**: trains both a Random Forest (RF) and a
   Gaussian Process (GP) on 70% of the data, predicts on the remaining 30%.
3. **(B) Repeated 5-fold CV**: runs `N_CV_FOLDS × N_CV_REPS` fits, averages
   the out-of-fold predictions.
4. Reports for each model × split: **Pearson r**, **Spearman ρ**, **RMSE**,
   **MAE**.
5. **Saves two plots:**
   - `s1_predicted_vs_true.png` — 4-panel predicted-vs-true scatter (RF
     hold-out, GP hold-out, RF CV, GP CV), coloured by dominant cation region.
   - `s1_residuals_by_region.png` — residual plots coloured by region to
     detect systematic bias.

The surrogate models used:
- **Random Forest**: 500 trees, `min_samples_leaf=2`
- **Gaussian Process**: `ConstantKernel × Matérn(ν=2.5) + WhiteKernel`,
  fitted with 5 restarts, `normalize_y=True`

#### §2 — Pool-based Virtual BO Experiment (`section2`)

1. Uses all unique compositions as a fixed **pool**.
2. Starts each run with `N_SEED_BO` random compositions.
3. At each of `N_BO_ITER` steps:
   - Fits a GP surrogate on all observed points.
   - Computes **Expected Improvement (EI)** on all unobserved pool points.
   - Picks the point with the highest EI.
   - "Measures" it by looking up the true value from the dataset.
4. Compares four strategies: **GP+EI**, **RF+EI**, **Greedy** (GP mean only,
   no exploration), **Random**.
5. Runs `N_BO_RUNS` independent repetitions for confidence bands.
6. Reports: median steps to reach the top-20% threshold for each strategy.
7. **Saves two plots:**
   - `s2_bo_convergence.png` — convergence curves (best found vs. evaluations).
   - `s2_bo_convergence_normalised.png` — normalised version (1.0 = global best found).

#### §3 — Acquisition Surface & Suggestion Sanity (`section3`)

1. Generates a regular grid of `~861` compositions covering the full simplex
   (step = `GRID_STEP`).
2. Predicts GP mean, GP std, and RF mean on the grid.
3. Computes EI across the grid.
4. Picks the **top-5 EI grid points** as "next suggestions" and checks:
   - Do the fractions sum to 1.0? ✓/✗
   - Are they inside the convex hull of the existing data (not extrapolating)?
   - Reports predicted mean ± σ for each.
5. Computes correlation between GP predicted mean and MA fraction (should be
   strongly positive — MA-rich = more unstable).
6. **Saves four plots:**
   - `s3_gp_mean.png` — ternary heatmap of GP predicted mean (log scale).
   - `s3_gp_std.png` — ternary heatmap of GP uncertainty.
   - `s3_gp_ei.png` — ternary heatmap of EI (acquisition surface), with
     gold stars marking the top-5 suggestions.
   - `s3_rf_mean.png` — ternary heatmap of RF predicted mean.

#### §4 — Domain Knowledge Cross-check (`section4`)

1. Predicts GP and RF values at the three pure-component corners (pure Cs,
   pure FA, pure MA).
2. Verifies that MA corner > Cs corner and MA corner > FA corner (literature:
   pure MAPbI₃ is unstable).
3. Prints the top-10 observed compositions (lowest mean instability).
4. Prints the top-10 grid compositions by GP predicted mean.
5. Checks that the best GP-predicted composition has MA < 20%.
6. **Saves two plots:**
   - `s4_top10_compositions.png` — ternary diagram with all data in grey and
     top-10 highlighted as diamonds (left: GP prediction; right: observed).
   - `s4_instability_vs_MA.png` — scatter of instability vs. MA fraction
     overlaid with GP mean curve from the grid.

#### §5 — Summary Pass/Fail Report (`section5`)

Prints a formatted table with 10 checks and their status. Also saves
`validation_results/surrogate_metrics.csv` with all numeric metrics in a
machine-readable format.

---

### 7.5 Output Files

All saved to `validation_results/`:

| File | What it shows |
|---|---|
| `s0_ternary_raw.png` | Raw data landscape — where stable compositions are |
| `s0_replicate_variance.png` | Experimental noise level across the triangle |
| `s1_predicted_vs_true.png` | Surrogate accuracy (4 panels) |
| `s1_residuals_by_region.png` | Systematic bias check by region |
| `s2_bo_convergence.png` | BO vs. random convergence curves |
| `s2_bo_convergence_normalised.png` | Same, normalised to global best |
| `s3_gp_mean.png` | GP predicted mean heatmap |
| `s3_gp_std.png` | GP uncertainty heatmap (where the model is least confident) |
| `s3_gp_ei.png` | Acquisition surface + top-5 suggested experiments (gold stars) |
| `s3_rf_mean.png` | RF predicted mean heatmap (cross-check vs. GP) |
| `s4_top10_compositions.png` | Where the model thinks the best compositions are |
| `s4_instability_vs_MA.png` | MA-fraction trend check |
| `surrogate_metrics.csv` | Tabular: Pearson r, Spearman ρ, RMSE, MAE for both models × both splits |

---

### 7.6 Interpreting the Pass/Fail Report

The final checklist uses these thresholds (all adjustable in `section5`):

| Check | Pass condition | What it means if it fails |
|---|---|---|
| Composition sums | 0 violations | Data has a formatting error — fix before using |
| Pearson r > 0.5 on hold-out | Both RF and GP | Model cannot predict held-out points — trust no suggestions |
| Spearman ρ > 0.5 on hold-out | Both RF and GP | Model cannot rank compositions correctly |
| CV RMSE < 30% of range | Both models | Prediction error is too large relative to the measurement scale |
| BO reaches top-20% faster than random | GP+EI ≤ Random | BO gives no benefit — check kernel, scaling, or acquisition settings |
| Top-5 EI suggestions inside convex hull | All 5 | Model is suggesting extrapolations — increase data coverage or reduce `GRID_STEP` |
| Top-5 suggestions sum to 1.0 | All 5 | Grid generation bug — never happens with the current simplex grid |
| GP mean vs. MA correlation r > 0.3 | Positive correlation | Model has inverted the physics — check target column direction |
| MA corner worse than Cs corner | GP predicts correctly | Model is biased — investigate outliers or kernel choice |
| Best GP composition has MA < 20% | True | Model recommends MA-rich compositions — physically wrong, check scaling |

**Overall rating:**
- 10/10 PASS → surrogate and BO are reliable for this dataset.
- 7–9/10 → review failures individually; minor issues may be acceptable.
- < 7/10 → something is fundamentally wrong; do not rely on suggestions.

---

## 8. Configuration Reference

### `StudyConfig` — full field table

| Field | Type | Default | Notes |
|---|---|---|---|
| `parameters` | `List[ParameterConfig]` | `[]` | Ordered list of all input parameters |
| `objectives` | `List[ObjectiveConfig]` | `[]` | 1 objective = single-obj; ≥2 = multi-obj |
| `batch_size` | `int` | `1` | Experiments per round (1–20 is typical) |
| `n_batches` | `int` | `10` | Total rounds; can be increased mid-session |
| `sampler_name` | `str` | `"TPE"` | `"TPE"`, `"NSGAII"`, or `"Random"` |

### `ParameterConfig` — full field table

| Field | Type | Default | Applies to |
|---|---|---|---|
| `name` | `str` | required | All |
| `ptype` | `ParameterType` | required | All |
| `enabled` | `bool` | `True` | All |
| `full_min` | `float` | `0.0` | INT, FLOAT |
| `full_max` | `float` | `1.0` | INT, FLOAT |
| `allowed_subranges` | `List[AllowedSubRange]` | `[full_min, full_max]` | INT, FLOAT |
| `step` | `Optional[int]` | `None` (= 1) | INT only |
| `all_choices` | `List[str]` | `[]` | CATEGORICAL, BOOL |
| `allowed_choices` | `List[str]` | `[]` | CATEGORICAL |
| `fixed_value` | `Optional[bool]` | `None` | BOOL only |

### Sampler comparison

| Sampler | Best for | Notes |
|---|---|---|
| `TPE` | Single objective, ≤ 20 parameters | Uses multivariate Parzen estimator. Best default choice. |
| `NSGAII` | Multi-objective | Genetic algorithm; batch sizes should be multiples of the population size |
| `Random` | Baselines / debugging | No learning; useful to verify the BO is adding value |

---

## 9. Common Adjustments & Recipes

### Change the dataset

1. Put your new CSV in `test_data/` (or anywhere accessible).
2. Launch the app → **File → Load CSV** → select your file.
3. The app auto-detects column types. Review the parameter cards and adjust
   ranges / types if needed.

---

### Add a dead region (forbidden parameter zone)

In the GUI:
1. Click the **Dead Regions** button on the parameter card.
2. Click **+** to add a row.
3. Enter the `[low, high]` of the forbidden interval.
4. Click **Apply** — the preview updates to show the remaining allowed ranges.

In code:
```python
from sampler_utils import compute_allowed_subranges
from parameter_config import ParameterConfig, ParameterType

config = ParameterConfig(
    name="spin_speed",
    ptype=ParameterType.INT,
    full_min=1000,
    full_max=5000,
    allowed_subranges=compute_allowed_subranges(1000, 5000, [(2800, 3200)]),
)
# Allowed: [1000, 2800] and [3200, 5000]
```

---

### Run multi-objective optimisation

1. In the Objectives panel, select **two or more** result columns.
2. Set direction for each (minimize / maximize).
3. The sampler is automatically switched to `NSGAII`.
4. After optimisation, the centre panel shows a **Pareto front** table instead
   of a single best trial.

In code:
```python
config = StudyConfig(
    objectives=[
        ObjectiveConfig("efficiency", "maximize"),
        ObjectiveConfig("cost",       "minimize"),
    ],
    sampler_name="NSGAII",
    ...
)
```

---

### Warm-start from a historical CSV

Load a CSV that already has result columns filled in. When you start the
optimisation, the app calls `load_historical_trials()` to inject all
complete rows into the study as COMPLETE trials. Optuna will use this history
to make better initial suggestions.

Deduplication is automatic — reloading the same CSV multiple times is safe.

---

### Change the BO sampler at runtime

In the GUI, change the **Sampler** dropdown before clicking "Ask Next Batch".
The new sampler takes effect from the next session creation — it does not
affect an already-running study since the sampler is fixed at `build_study()`
time.

To switch sampler mid-study programmatically, create a new study with
`load_if_exists=False` (losing history) or use `study.sampler =
optuna.samplers.TPESampler()` directly (undocumented but works).

---

### Adjust the Expected Improvement exploration level (validation script)

In `validate_perovskite.py`, change `EI_XI`:

```python
EI_XI = 0.01   # default — moderate exploration
EI_XI = 0.0    # pure exploitation (greedy)
EI_XI = 0.1    # more exploration (useful when the landscape is noisy)
```

---

### Point the validation script at a different dataset

1. Change `CSV_PATH`, `FEATURES`, and `TARGET` at the top of
   `validate_perovskite.py`.
2. If your data is not a 3-component ternary, the ternary plot functions will
   need to be replaced — they assume exactly 3 feature columns.
   For non-ternary data, replace `draw_ternary_frame` and `ternary_to_cart`
   with standard 2-D scatter or parallel-coordinates plots.

---

### Run only a subset of validation sections

Each section is a standalone function. You can call only what you need:

```python
from validate_perovskite import section0, section1
import pandas as pd

df = pd.read_csv("test_data/Perovskite_dataset.csv")
s0 = section0(df)
s1 = section1(df)
# Skip §2 (slow virtual BO) and go straight to domain check
```

---

## 10. Troubleshooting

### `UnicodeEncodeError: 'charmap' codec can't encode character`

**Cause:** Windows terminal using cp1252 encoding; the script prints Unicode
characters (─, ✓, ρ, §, etc.).

**Fix:** The script already calls `sys.stdout.reconfigure(encoding="utf-8")`
at startup. If you still see this error, run:

```cmd
chcp 65001
python validate_perovskite.py
```

Or launch from Windows Terminal (which defaults to UTF-8) instead of the
legacy cmd.exe.

---

### `FileNotFoundError: Optuna database not found`

**Cause:** The `_session.json` was moved without its paired `.db` file.

**Fix:** Keep both files in the same directory. If the `.db` was deleted,
the session cannot be recovered — start a new session and reload your CSV.

---

### `The system cannot find the file python.exe`

**Cause:** The Windows Store python stub is on PATH but points nowhere.

**Fix:** Use the full path to your Python interpreter:
```cmd
"C:\Users\<you>\AppData\Local\Python\pythoncore-3.14-64\python.exe" main.py
```
Or configure PyCharm's run configuration to use the correct interpreter
(`Settings → Project → Python Interpreter`).

---

### Session shows "pending batch" on startup but no dialog appears

**Cause:** The app was closed while a batch was pending. On the next start,
the session is loaded with `pending_batch` set.

**Fix:** Use **File → Load Updated CSV** to load a CSV that contains the
results, then the app will auto-match and open the result dialog. Alternatively,
click "Ask Next Batch" again — the pending batch results dialog will open
automatically.

---

### Optuna suggests values outside an expected range

**Cause:** The `allowed_subranges` may not be set correctly, or the type
was inferred as FLOAT when INT was expected.

**Fix:** Check the parameter card for that column. Ensure the type is correct
and the range is as expected. For integer parameters, make sure the CSV
column has no decimal values (e.g. `2000` not `2000.0`).

---

### The GP in the validation script is very slow

The GP fits `n_restarts_optimizer=5` times per fit. With 25 CV folds × 2 models
+ 10 BO runs × 30 steps, this adds up.

**To speed up:**
```python
# In validate_perovskite.py
N_CV_REPS  = 2   # reduce from 5
N_BO_RUNS  = 5   # reduce from 10

def make_gp():
    ...
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=2,   # reduce from 5
        normalize_y=True, random_state=SEED,
    )
```

---

*End of Documentation*
