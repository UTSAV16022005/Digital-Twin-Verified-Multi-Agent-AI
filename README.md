# Digital-Twin-Verified Multi-Agent AI for Aerospace Health Management

This project develops a reproducible research prototype for aerospace fault monitoring, diagnosis, prognostics, recovery planning, and safety verification.

## Current scope

The first concrete system is a simulated turbofan engine using NASA's C-MAPSS data.

- FD001 is the first end-to-end baseline because it has one operating condition and one fault mode.
- FD004 is the final benchmark because it has six operating conditions and two fault modes.
- Recovery actions are not learned directly from C-MAPSS. We will define an explicit action space and a transparent Digital Twin transition model for what-if simulation.

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

The core project runs without PyTorch using its deterministic NumPy LSTM
implementation. PyTorch is optional and gives the agent a faster native backend.
Install the core requirements and, where permitted, the CPU PyTorch wheel:

```powershell
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
python -m scripts.train_lstm_monitoring
```

The official PyTorch installer supports Windows and provides a CPU-only wheel;
see [PyTorch Start Locally](https://pytorch.org/get-started/locally/). This
machine currently blocks `torch_python.dll` with Windows Application Control
(WinError 4551). The training script records that import diagnostic and runs
the NumPy LSTM backend instead; changing packages or using a virtual environment
does not bypass an operating-system application-control policy. Use a permitted
machine or environment for the PyTorch backend.

## Diagnosis Agent

The first diagnosis implementation classifies project-defined health states
(`healthy`, `degraded`, `critical`) from RUL-derived labels. FD001 contains one
degradation mode and does not provide a direct component-fault label, so this
baseline does not claim physical fault identification. True fault-type diagnosis
will use FD004 and controlled fault-injection cases.

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

The formal `PrognosticsAgent` currently uses a Random Forest RUL baseline with
ensemble-dispersion uncertainty. It exposes the same contract that a future
LSTM/Transformer prognostics model can implement. The integrated demo passes the
predicted RUL into the Digital Twin; ground-truth RUL is recorded only for
evaluation and is not used by the Twin or planner.

```powershell
python -m scripts.train_prognostics_agent
python -m scripts.demo_agent_pipeline
```

## Safety Verification and closed loop

The safety layer independently checks every Twin outcome against hard limits on
failure risk, model uncertainty, projected drift, Twin constraint violations,
and shutdown policy. The closed-loop demo passes Monitoring, Diagnosis,
Prognostics, Digital Twin, Planning, and Safety results through one auditable
decision step.

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
```

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

