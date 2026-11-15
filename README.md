# Digital-Twin-Verified Multi-Agent AI for Aerospace Health Management

This project develops a reproducible research prototype for aerospace fault monitoring, diagnosis, prognostics, recovery planning, and safety verification.

## Current scope

The first concrete system is a simulated turbofan engine using NASA's C-MAPSS data.

- FD001 is the first end-to-end baseline because it has one operating condition and one fault mode.
- FD004 is the final benchmark because it has six operating conditions and two fault modes.
- Recovery actions are not learned directly from C-MAPSS. The prototype uses an explicit action space and a transparent Digital Twin transition model for what-if simulation; action effects remain assumptions.

## Architecture implementation

1. **Data preprocessing**: validation, sensor selection, normalization, feature engineering, and time windows.
2. **Monitoring agent**: healthy-state reconstruction and anomaly scoring.
3. **Diagnosis agent**: degradation or fault-state classification with an interpretable baseline.
4. **Prognostics agent**: Remaining Useful Life prediction with a sequence model.
5. **Recovery planning agent**: candidate action generation and ranking.
6. **Digital Twin**: current-state estimation and next-state simulation under candidate actions.
7. **Safety verifier**: deterministic constraints that reject unsafe actions.
8. **Closed-loop evaluation**: telemetry update, state update, twin update, and repeat.

## Research quality rules

- Split by engine unit, never by random rows.
- Fit preprocessing transforms on training units only.
- Keep a simple baseline before adding deep learning or reinforcement learning.
- Report metrics, uncertainty, failure cases, and ablations.
- Treat the safety verifier as a hard constraint layer, not as an LLM opinion.

## First milestone

Run the following commands and confirm the raw data before creating training windows:

```powershell
python -m scripts.audit_cmapss
python -m scripts.profile_fd001
python -m scripts.train_rul_baseline
python -m scripts.evaluate_fd001_test
python -m scripts.run_monitoring_baseline
```

## LSTM monitoring agent

The project contains the architecture-aligned LSTM autoencoder in
`src/lstm_monitoring.py`. It trains only on early-life healthy windows, fits its
scaler on fit engines, selects the checkpoint and calibrates its threshold on a
separate calibration-engine group, and reports on held-out validation engines
and the official test engines. It confirms alerts after persistent
above-threshold windows. The PCA implementation remains a separate classical
baseline for comparison.

The monitoring agent supports a CPU PyTorch backend and a deterministic NumPy
fallback. The current environment successfully trained the LSTM autoencoder
with PyTorch 2.15.0+cpu. Install the core requirements and, where permitted,
the CPU PyTorch wheel:

```powershell
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
python -m scripts.train_lstm_monitoring
```

The latest CPU PyTorch run detected degradation on 84.8% of the 33 official
test engines that had degraded windows, with 4.36% healthy-window false alarms
and 65-cycle median lead time. Validation engine detection was 100% with a
52-cycle median lead time. On the same split and alert protocol, PCA detected
2% of those test engines. These are C-MAPSS anomaly-monitoring results, not
physical fault diagnosis or evidence of deployment safety. The run report is
`reports/generated/lstm_monitoring_verified/fd001_lstm_monitoring.json`.

The official PyTorch installer supports Windows and provides a CPU-only wheel;
see [PyTorch Start Locally](https://pytorch.org/get-started/locally/). If native
libraries are restricted in another environment, the training script records
the import diagnostic and falls back to the NumPy implementation.

## Diagnosis Agent

The integrated diagnosis agent classifies project-defined health states
(`healthy`, `degraded`, `critical`) from RUL-derived labels. FD001 and FD004 do
not provide sensor-level physical fault labels, so this agent does not claim
component fault identification. A separate experimental classifier recognizes
only controlled, project-generated sensor corruptions; its labels are synthetic
and are not physical fault labels.
The health-state learner applies inverse-frequency sample weights so the
project-defined healthy, degraded, and critical classes contribute evenly.
The synthetic corruption experiment also generates balanced class counts and
reports per-class precision and recall.

The project thresholds are healthy when RUL > 125 cycles, degraded when
25 < RUL <= 125, and critical when RUL <= 25. These are project-defined labels
derived from RUL, not NASA-provided health or physical fault labels.

```powershell
python -m scripts.train_diagnosis_baseline
```

## Digital Twin prototype

The first Digital Twin learns a one-step telemetry transition from training
engines and simulates explicit action assumptions for what-if analysis. It does
not claim to reproduce a physical engine or learn recovery effects from C-MAPSS,
because C-MAPSS contains no action/outcome data. A later planner will consume
these outcomes, and a separate safety layer will enforce mission constraints.

```powershell
python -m scripts.demo_digital_twin
```

## Recovery Planning Agent prototype

The first planner simulates the registered actions through the Digital Twin,
rejects outcomes that violate hard constraints or exceed the configured risk
budget, and ranks the remaining actions with an auditable weighted objective.
It is a constrained planner, not a PPO policy yet. PPO can be evaluated later
after the Twin has a validated transition environment.

```powershell
python -m scripts.demo_recovery_planner
```

## Prognostics Agent and integrated pipeline

The formal `PrognosticsAgent` defaults to a Random Forest RUL model and also
supports XGBoost, Ridge, a naive baseline, NumPy LSTM, PyTorch CNN, and PyTorch
Transformer models. Its uncertainty bounds are calibrated on held-out engines.
The integrated demo passes predicted RUL into the Digital Twin; ground-truth
RUL is recorded only for evaluation and is not used by the Twin or planner.

```powershell
python -m scripts.train_prognostics_agent
python -m scripts.demo_agent_pipeline
```

To train a sequence-model option with the same agent interface, use
`python -m scripts.train_prognostics_agent --model-type lstm` (or choose
`pytorch_cnn` or `pytorch_transformer`). For model selection, use the grouped
training-only experiment below; it compares all six model candidates and the
cycle-feature RF across five engine-disjoint folds.

## Safety Verification and closed loop

The safety layer independently checks every Twin outcome against hard limits on
failure risk, model uncertainty, projected drift, Twin constraint violations,
shutdown policy, and basic validity of the predicted state, RUL, horizon, and
mission cost. It also rejects invalid verifier thresholds at configuration
time. The closed-loop demo passes Monitoring, Diagnosis, Prognostics, Digital
Twin, Planning, and Safety results through one auditable decision step.
Prognostics uncertainty is calibrated on held-out development engines; the
nominal RUL and symmetric conformal interval are reported, while the Twin plans
from a one-sided lower 95% bound calibrated on overestimation residuals. The
orchestrator abstains with no action when telemetry is shorter than its
configured window or uncertainty has not been calibrated.

```powershell
python -m scripts.demo_closed_loop
```

## Adversarial safety tests and FD004 evaluation

Safety tests cover high failure risk, high uncertainty, excessive projected
drift, Twin-reported violations, and a disabled-shutdown policy. FD004 is the
harder C-MAPSS subset with multiple operating conditions and degradation modes;
its evaluation remains health-state and prognostics-focused because public
fault-type and action-outcome labels are unavailable.

```powershell
python -m unittest discover -s tests -v
python -m scripts.evaluate_fd004_pipeline
python -m scripts.evaluate_safety_verifier
```

The Safety Verifier evaluation checks a fixed set of synthetic boundary and
malformed-input scenarios. It measures whether the deterministic rules behave
as specified; it is not an empirical safety rate and does not establish that
the simulated Twin or its actions are physically safe. It writes
`safety_verifier_scenarios.json` and `safety_verifier_scenarios.csv` under
`reports/generated/`.

## Closed-loop development evaluation

To evaluate decisions across held-out engines and fixed telemetry milestones,
run:

```powershell
python -m scripts.evaluate_closed_loop_development
```

This uses FD001 and FD004 training trajectories only, split by engine into
60% fitting, 20% uncertainty calibration, and 20% decision evaluation. The
calibration group has at least 19 engines, enough for a finite 95% conformal
order statistic. Because calibration points are selected across RUL ranges,
empirical coverage is also checked separately at held-out cycle milestones;
the quantile alone is not a guarantee for every operating regime. The
evaluation reports recommendations,
safety rejections, abstentions, action counts, and
RUL errors at cycles 10, 20, 30, 60, 90, and 120. Actual RUL labels are withheld
from agents and used after each decision for scoring. Repeated-milestone rate
and RUL intervals resample engines as clusters; single-milestone rates use
Wilson intervals across independent engines. The JSON summary and state-level
CSV are saved under `reports/generated/`.
This describes the prototype's behavior under its configured rules; C-MAPSS has
no recovery-action outcomes, so it does not validate the Digital Twin's action
effects or establish operational safety. The Twin models controlled shutdown
as a frozen, zero-risk state and the current policy permits it, so a zero
`no_safe_action` rate reflects that built-in fallback and is not evidence that
every operating action is safe.

To diagnose FD004 RUL errors and identify the calibration residual setting the
planning margin, run:

```powershell
python -m scripts.analyze_fd004_rul_errors
```

This reuses the saved fit/calibration/evaluation engine split, reports errors by
cycle, actual-RUL band, and operating-setting cluster, and writes the selected
calibration residual for each calibration engine. It does not load official
test files. Outputs are saved as `fd004_rul_error_diagnostics.json` and related
CSV files under `reports/generated/`.

To compare the 30-cycle RF, 40-cycle RF, operating-condition-normalized RF,
and NumPy LSTM without selecting on the calibration or final evaluation engines,
run:

```powershell
python -m scripts.compare_fd004_prognostics_models
```

This performs five engine-grouped folds using only the fit-engine group. It
compares common fixed-cycle milestones and reports engine-clustered confidence
intervals. Outputs are saved as `fd004_model_selection_cv.json`,
`fd004_model_selection_metrics.csv`, and
`fd004_model_selection_oof_predictions.csv` under `reports/generated/`.

To test whether the FD004 life-stage bias can be corrected without using the
reserved calibration or final evaluation engines, run:

```powershell
python -m scripts.evaluate_fd004_bias_calibration
```

This uses nested engine-grouped folds within the fit-engine group. An isotonic
mapping is learned from inner out-of-fold predictions and assessed on outer
held-out engines at fixed cycle milestones. It is an experiment and does not
change the production prognostics model. Results are written to
`fd004_bias_calibration_cv.json` and related CSV files under `reports/generated/`.

To test an explicit engine-age input against the same RF30 baseline using only
fit-engine grouped folds, run:

```powershell
python -m scripts.evaluate_fd004_cycle_age_feature
```

The evaluation compares fixed-cycle out-of-fold predictions and keeps the
calibration and final evaluation engine groups reserved. Outputs are saved as
`fd004_cycle_age_feature_cv.json` and
`fd004_cycle_age_feature_oof_predictions.csv`.

To compare this candidate once on the previously reserved calibration and
decision-evaluation engines, run:

```powershell
python -m scripts.evaluate_fd004_cycle_age_holdout
```

The cycle feature reduced decision-state MAE from 58.60 to 54.25 cycles on the
50 held-out FD004 engines, but the paired engine-level 95% confidence interval
for the improvement includes zero (−8.872 to +0.001 cycles). Planning lower-bound
coverage also fell from 98.5% to 92.0%. Therefore
`PrognosticsConfig.include_cycle_feature` is opt-in and remains `False` by
default after this single holdout comparison. Outputs are saved as
`fd004_cycle_age_holdout.json` and `fd004_cycle_age_holdout_states.csv`.

To explore age-conditioned safety margins for the cycle-feature model, run:

```powershell
python -m scripts.evaluate_fd004_cycle_conditioned_calibration
```

This estimates a separate one-sided 95% overestimation margin at cycles 30, 60,
90, and 120 using the calibration engines; other cycles use the pooled margin.
On the already-inspected development holdout, observed coverage at those
milestones changed from 96%, 86%, 90%, and 96% with the pooled margin to 98%,
92%, 100%, and 98% with milestone margins. RUL predictions and MAE did not
change; the planning bounds became more conservative, increasing simulated
controlled-shutdown recommendations from 10 to 41. Cycle 60 remained below the
nominal 95% coverage level, so this remains an exploratory, opt-in calibration
mode rather than a validated safety guarantee. The experiment reuses the prior
development holdout and is not an independent confirmation. Outputs are saved
as `fd004_cycle_conditioned_calibration.json` and
`fd004_cycle_conditioned_calibration_states.csv`.

To estimate the cycle-feature and milestone-margin results across engine groups,
run:

```powershell
python -m scripts.evaluate_fd004_cycle_conditioned_cv
```

This uses five outer engine-group folds. Each fold separates approximately
60% of engines for fitting, 20% for margin calibration, and 20% for evaluation;
every engine appears in an evaluation fold once. Across FD004, RF30 MAE was
54.82 cycles and RF30 with cycle age was 48.44 cycles. The paired engine-level
MAE improvement was 6.38 cycles (95% CI 4.38–8.34). Milestone margins raised
engine-averaged lower-bound coverage from 92.5% to 95.2%; the paired increase
was 2.7 percentage points (95% CI 1.6–4.0). Per-cycle coverage with milestone
margins was 94.4%, 95.6%, 95.6%, and 95.2% at cycles 30, 60, 90, and 120. This
evaluates RUL and its lower bound, not Digital Twin action outcomes. Outputs are saved as
`fd004_cycle_conditioned_cv.json` and
`fd004_cycle_conditioned_cv_predictions.csv`.

The same grouped evaluation on FD001 is available with:

```powershell
python -m scripts.evaluate_fd001_cycle_conditioned_cv
```

Across its 100 engines, cycle age reduced MAE from 39.52 to 34.84 cycles; the
paired improvement was 4.68 cycles (95% CI 2.99–6.35). Pooled-margin coverage
was 96.8%, while milestone-specific margins gave 95.0%; their paired coverage
difference was −1.75 percentage points (95% CI −4.25 to +0.50). Milestone
coverage on FD001 was 95%, 99%, 93%, and 93% at cycles 30, 60, 90, and 120.
This supports the cycle feature as an accuracy candidate on both subsets, while
showing that the milestone-margin benefit seen on FD004 does not carry over to
FD001. Keep the feature opt-in and choose calibration by subset only after the
final evaluation protocol is locked. Results are saved as
`fd001_cycle_conditioned_cv.json` and
`fd001_cycle_conditioned_cv_predictions.csv`.

The FD004 evaluation saves `reports/generated/fd004_terminal_engine_errors.csv`
with one held-out terminal prediction per scored engine. `signed_error` is
predicted RUL minus actual RUL, so a positive value means the model
overestimated remaining life. The JSON report gives the scored-engine count;
engines shorter than the 30-cycle input window are listed in
`reports/generated/fd004_unscored_test_engines.csv` and are excluded from the
terminal metrics. Use these reports to inspect errors and coverage; do not tune
model choices against the official test results.

## Grouped validation and ablations

For repeatable model comparisons, run the engine-level cross-validation and
feature/window/model ablation suite:

```powershell
python -m scripts.run_research_evaluation
```

This uses five `GroupKFold` folds on FD001 training engines and a separate
engine-level FD004 development split for operating-condition normalization
ablations. Scalers and feature selection are fit inside each fold. The official
test files are not loaded by this experiment. JSON summaries and per-engine CSV
outputs are saved under `reports/generated/`.

## Short-history and overestimation experiment

To study inputs shorter than the standard 30-cycle window, run:

```powershell
python -m scripts.evaluate_prognostics_robustness
```

This fits a variable-window model and fixed-window references on FD004 fit
engines, then evaluates 10–30-cycle prefixes on separate development engines.
It also measures a median-bias correction using engine-level cross-fitting. The
official test files are not loaded, and the correction is not applied to the
production agent. In the current run, raw prefix MAE ranged from about 63.6 to
68.0 cycles; the cross-fitted correction still ranged from about 55.6 to 61.9.
That is not reliable enough to use for decisions, so the main agent retains its
30-cycle minimum and abstains when history is shorter. Outputs are
`fd004_variable_history_summary.json` and
`fd004_variable_history_predictions.csv` under `reports/generated/`.

## Integrated LSTM prognostics and agent coordination

Install the optional XGBoost comparison dependency, then run the five-fold
engine-grouped comparison of the integrated PrognosticsAgent models:

```powershell
python -m pip install -r requirements-model-comparison.txt
python -m scripts.evaluate_prognostics_agent_models_cv
```

This compares RF, XGBoost, RF with engine-cycle age, the integrated NumPy LSTM,
a temporal CNN, and a Transformer. Each outer fold uses the same stride-5
windows and engine groups: approximately 51% for fitting, 9% for neural-model
early stopping, 20% for uncertainty calibration, and 20% for evaluation.
Preprocessing is fit on model-fit engines only. All prediction metrics and
paired comparisons use out-of-fold results from C-MAPSS training trajectories;
the official test files are not loaded.

| Model | FD001 MAE (cycles) | FD001 paired change vs RF (95% CI) | FD004 MAE (cycles) | FD004 paired change vs RF (95% CI) |
|---|---:|---:|---:|---:|
| RF | 37.31 | baseline | 54.67 | baseline |
| XGBoost | 36.96 | −0.35 [−1.42, 0.73] | 54.37 | −0.30 [−1.26, 0.67] |
| RF + cycle age | **33.42** | **−3.89 [−5.44, −2.31]** | **48.69** | **−5.98 [−7.75, −4.18]** |
| NumPy LSTM | 37.79 | +0.48 [−1.28, 2.23] | 54.18 | −0.48 [−2.52, 1.53] |
| Temporal CNN | 38.50 | +1.19 [−0.72, 3.08] | 55.79 | +1.12 [−1.36, 3.58] |
| Transformer | 38.55 | +1.24 [−0.58, 3.05] | 53.15 | −1.51 [−3.62, 0.62] |

The cycle-age RF has the clearest MAE improvement on both subsets, but its
FD004 calibrated lower-bound coverage is 93.4%, below the nominal 95% target;
it stays opt-in. The XGBoost, LSTM, CNN, and Transformer paired intervals include
zero on both subsets, so this run does not establish that they outperform the
standard RF. These are fixed-config model comparisons, not exhaustive tuning or
independent external validation. Reports include per-cycle metrics, per-engine
predictions, early-stopping epochs, and paired engine-bootstrap confidence
intervals under `reports/generated/`.

The role coordinator connects Monitoring, Diagnosis, Prognostics, the Twin,
Recovery Planning, and Safety Verification and writes a step-by-step trace:

```powershell
python -m scripts.demo_closed_loop
python -m scripts.evaluate_closed_loop_candidate_models
```

The closed-loop demo defaults to PCA for a classical monitoring comparison. To
run the same coordinator with the trained LSTM autoencoder monitor, use:

```powershell
python -m scripts.demo_closed_loop --monitor-backend lstm
```

The coordinator adapts both monitor outputs to a common anomaly score and alert
trace; the LSTM path uses the model saved by `train_lstm_monitoring`. The PCA
and LSTM demos save separate JSON traces:
`fd001_closed_loop_demo.json` and `fd001_closed_loop_demo_lstm.json`.

The integrated demo can run the same role sequence on any included C-MAPSS
subset (`FD001` through `FD004`) using the `--subset` option. For example, PCA
monitoring on FD002 is:

```powershell
python -m scripts.demo_closed_loop --subset FD002
```

To use the LSTM monitor on FD002, train and save a separate subset-specific
monitor first, then run the integrated demo:

```powershell
python -m scripts.train_lstm_monitoring --subset FD002 --model-dir models/fd002_lstm_monitoring_verified --report-dir reports/generated/fd002_lstm_monitoring_verified
python -m scripts.demo_closed_loop --subset FD002 --monitor-backend lstm
```

The diagnosis, prognostics, Twin, planner, and safety components are fitted or
configured from that subset for the demo. Output metrics and the selected
action can differ from FD001; use each subset's trace as a separate result.

The coordinator accepts telemetry only and rejects an input frame containing
the `rul` target. Diagnosis outputs are explicitly marked as RUL-derived health
bands. This is a deterministic, role-based agent system with auditable handoffs;
it is not an LLM or open-ended autonomous system. The closed-loop candidate
comparison reuses an already inspected development split and must be read as
development evidence. Its JSON and state-level results are written to
`reports/generated/closed_loop_candidate_models.json` and
`reports/generated/closed_loop_candidate_models_states.csv`.

## Synthetic sensor-fault experiment

To run the separate sensor-corruption classifier experiment, use:

```powershell
python -m scripts.evaluate_synthetic_fault_diagnosis
```

The experiment splits by engine and injects synthetic bias, drift, stuck-value,
and spike patterns into telemetry windows. It does not use the official C-MAPSS
test files. Current held-out-engine accuracy is 73.3% on FD001 and 81.4% on
FD004; macro-F1 is 0.743 and 0.816. Those metrics measure recognition of the
same documented family of generated corruptions under different held-out
engines and severity ranges. They do not measure diagnosis of physical engine
faults. The report includes per-class performance, calibration metrics,
confusion matrices, and engine-cluster bootstrap intervals in
`reports/generated/synthetic_fault_diagnosis.json`.

## Digital Twin validation and action sensitivity

Validate nominal next-cycle sensor prediction against persistence using
five-fold engine-grouped cross-validation:

```powershell
python -m scripts.evaluate_digital_twin_transition_cv
```

The current run reduces normalized sensor RMSE from 0.744 to 0.548 on FD001
(100 held-out engines) and from 1.415 to 0.999 on FD004 (249 held-out engines).
This checks one-step predictions along observed no-action trajectories. It does
not validate action counterfactuals or long-horizon simulation.

To see how assumptions about action benefits affect recommendations, run:

```powershell
python -m scripts.analyze_action_assumption_sensitivity
```

This compares no-benefit, telemetry-only, degradation-only, configured, and
amplified action-effect assumptions on the existing development engine split.
It reports action counts and how often recommendations change. C-MAPSS contains
no action/outcome data, so these results quantify planner sensitivity to its
assumptions and cannot establish that an action is effective or safe in a real
engine. In the current run, removing both assumed operating benefits changed
45.0% of FD001 and 55.7% of FD004 recommendations versus the configured case;
changing only the degradation assumption changed 44.2% and 55.3%. The JSON and
state-level CSV are saved under `reports/generated/`.

## External dataset ingestion (SECOM and PHM 2024)

These datasets are added through task-specific adapters and a shared
`DatasetBundle` interface. Their raw feature meanings and task labels are
preserved; they are not renamed into C-MAPSS sensors, converted into RUL, or
passed to the C-MAPSS Digital Twin and recovery planner. This lets the project
compare suitable models per dataset without claiming the datasets describe the
same system.

- **UCI SECOM** is a semiconductor manufacturing pass/fail classification
  dataset with 1,567 rows and 590 numeric columns in `secom.data`. The UCI
  catalog describes 591 total fields in its combined dataset summary; the
  downloaded measurement file itself parses to 590 columns, with labels and
  timestamps supplied separately. The source labels `-1` (pass)
  and `1` (fail) are mapped to `faulty=0` and `faulty=1`. Timestamps and source
  labels stay in metadata. Missing measurements are preserved for preprocessing
  fitted on training data only. The dataset is highly imbalanced, so failure
  recall, F1, PR-AUC, and balanced accuracy matter more than accuracy alone.
- **PHM Society 2024 challenge** contains helicopter-turbine observations and
  supports two separate tasks: faulty/not-faulty classification and torque
  margin regression. Training has seven physical measurements and both labels.
  Test and validation provide measurements only in the downloaded files, so
  their targets are not invented. The supplied `id` is retained as sample
  metadata and excluded from model features. The challenge documentation says
  training observations were shuffled and asset IDs removed, so this project
  cannot claim asset-disjoint local validation from the supplied row IDs.
- **C-MAPSS** remains the project's turbofan sequence and RUL benchmark. Its
  sequence models, Digital Twin, and recovery-planning evaluation are specific
  to that task and are not automatically applied to SECOM or PHM 2024.

Download and extract source data under `data/raw/` using the official sources:

- [UCI SECOM dataset and description](https://archive.ics.uci.edu/dataset/179/secom)
- [PHM Society 2024 Data Challenge](https://data.phmsociety.org/phm2024-conference-data-challenge/)

Expected local layout:

```text
data/raw/secom/secom.data
data/raw/secom/secom_labels.data
data/raw/phm2024/extracted/training/X_train.csv
data/raw/phm2024/extracted/training/y_train.csv
data/raw/phm2024/extracted/test/X_test.csv
data/raw/phm2024/extracted/validation/X_validation.csv
```

Run a data-only audit before any model work:

```powershell
python -m scripts.audit_external_datasets
```

The audit checks schema, row counts, missing/infinite values, duplicate
features, label counts, feature-target ID alignment, and first/last five-row
previews. For PHM 2024 it also checks archive integrity and confirms the
extracted CSV files have the same SHA-256 hashes as the CSV members in the
archives. It does not fit models or modify the raw data. Raw archives and
extracted files remain ignored by Git; only adapters, audit code, and
documentation are committed.

The initial audit found 41,951 missing cells in SECOM and 116 constant SECOM
measurements. PHM 2024 has 742,625 labeled training rows, 21,436 test rows, and
21,436 validation rows; its downloaded test and validation files have no
targets. The PHM training split contains repeated feature rows, which are
reported but retained pending a documented train-only duplicate ablation.
These counts describe dataset integrity, not predictive performance.

## Implementation status and limits

Implemented end-to-end for this benchmark: preprocessing; anomaly monitoring;
RUL-band diagnosis; RF and LSTM prognostics; recovery candidate generation and
ranking; a learned nominal Twin transition; independent deterministic safety
checks; a telemetry-only coordinator; and closed-loop evaluation. Reports cover
engine-grouped validation, uncertainty, model comparisons, and sensitivity.

Still outside what these public labels can establish: physical component-fault
diagnosis, learned action effects, long-horizon counterfactual accuracy,
operational aerospace safety, or an RL policy trained against real outcomes.
The current planner is a transparent constrained heuristic, not PPO. The
current model comparison covers RF, XGBoost, RF with engine age, NumPy LSTM,
CNN, and Transformer prognostics. Those comparisons use fixed configurations;
broader hyperparameter search, physical fault labels, measured action effects,
and independent external validation remain future research work.

