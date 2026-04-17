# Experiment Report: [Short Title]

> **Author(s)**: [Names]
> **Date**: YYYY-MM-DD
> **Status**: [Completed | In Progress | Failed]

## 0. Design

_What was planned before the experiment ran? This section captures intent, not outcomes._

### 0.1 Evolution Target

_What function, algorithm, or code is being evolved? What are its inputs and outputs?_

### 0.2 Search Space

| Element | Mutable / Fixed | Details |
|---|---|---|
| _e.g. scorer function body_ | Mutable | _Python function returning dict[str, float]_ |
| _e.g. weight parameters_ | Mutable | _3 floats: KV, queue, evolved_ |
| _e.g. existing scorers_ | Fixed | _7 hand-written scorers unchanged_ |
| _e.g. gateway architecture_ | Fixed | _filter → score → select pipeline_ |
| ... | ... | ... |

### 0.3 Constraints

_Runtime limits, language restrictions, available libraries, timeout budgets, interface contracts._

### 0.4 Success Criteria

_Quantitative target defined before seeing results. e.g., "5–15% composite improvement" or "lower combined P99 vs baseline."_

## 1. Objective

_What question does this experiment answer? One paragraph max._

## 2. Hypothesis

_What did you expect to happen, and why?_

## 3. Setup

### 3.1 Hardware

| Resource | Value |
|---|---|
| Compute type | _e.g. GPU / CPU / TPU / FPGA_ |
| Compute units | _e.g. 9x NVIDIA A100-80GB / 64 CPU cores / 4x TPU v4_ |
| Memory | _e.g. 80 GB per GPU / 256 GB RAM_ |
| Platform | _e.g. OpenShift / LSF / AWS / bare-metal / local machine_ |
| Nodes | _e.g. 7 nodes, 4 GPU each / single workstation_ |
| Network | _e.g. InfiniBand / Ethernet / N/A_ |

### 3.2 Software

| Component | Version / Details |
|---|---|
| _Main software_ | _e.g. vLLM v0.7.2 / GROMACS 2024.1 / PyTorch 2.3_ |
| _Model (if applicable)_ | _e.g. Llama-3.1-8B-Instruct / N/A_ |
| _Benchmark tool_ | _e.g. inference-perf / MLPerf / custom script_ |
| _Orchestration / framework_ | **[REQUIRED]** repo URL · branch · commit SHA — e.g. `github.com/org/repo`, branch: `main`, commit: `abc1234` |
| _External simulator / tool_ | **[REQUIRED if used]** repo URL · commit SHA · clone path — e.g. `git clone https://github.com/org/tool.git path/to/tool` |
| _Other_ | _add rows as needed_ |

### 3.3 Configuration

_List every tunable parameter someone would need to reproduce the experiment._

| Parameter | Value | Rationale |
|---|---|---|
| _e.g. batch size_ | _32_ | _Largest that fits in memory_ |
| _e.g. load rate_ | _35 RPS_ | _Saturates the system without mass failures_ |
| ... | ... | ... |

### 3.4 Workload

_Describe the input data or traffic pattern. What requests/inputs? How generated? Why this workload?_

### 3.5 Baseline

_What is the control condition? How was it measured?_

| Metric | Baseline value |
|---|---|
| _e.g. throughput_ | _24 RPS_ |
| ... | ... |

## 4. Methodology

_Step-by-step description of how the experiment was executed. Enough detail that someone else could re-run it._

1. Step 1
2. Step 2
3. ...

### 4.1 Scoring / Evaluation

_How are results scored or evaluated? Formula, thresholds, metrics, acceptance criteria._

### 4.2 Reproducibility Checklist

- [ ] Framework repo URL, branch, and commit SHA recorded in Section 3.2
- [ ] External dependencies documented: repo URL, commit SHA, and clone location for each (e.g. `git clone https://github.com/org/tool.git path/to/tool`)
- [ ] Dependency lock file committed (e.g. `uv.lock`, `requirements.txt`, `go.sum`)
- [ ] All config files committed to repo (path: `...`)
- [ ] Baseline code / starting state documented or committed (path: `...`)
- [ ] Random seeds fixed where applicable
- [ ] Deployment scripts / launch commands available
- [ ] Data files (workload, inputs) available at `...`
- [ ] Environment variables and external service credentials documented (names only, not values)
- [ ] Manual reproduction commands verified against the actual run harness (e.g. `simulator.py`, `Makefile`) — not inferred from docs (see Appendix E)

## 5. Results

### 5.1 Summary

| Metric | Baseline | Best / Final | Change |
|---|---|---|---|
| _e.g. throughput_ | _24 RPS_ | _24.8 RPS_ | _+3.3%_ |
| ... | ... | ... | ... |

### 5.2 Detailed Results

_Charts, tables, distributions, per-component breakdowns, generation-over-generation plots, etc._

### 5.3 Failure Modes

_What went wrong? Quantify._

| Failure type | Count | Cause |
|---|---|---|
| _e.g. timeout_ | _6_ | _Worker overloaded_ |
| ... | ... | ... |

## 6. Analysis

### 6.1 Key Findings

1. Finding 1
2. Finding 2

### 6.2 Limitations

_What can't you conclude from this experiment?_

### 6.3 Comparison to Prior Work

_How does this compare to previous experiments or known baselines?_

### 6.4 Exploration Insights

_What did the search process reveal — strategies that failed, parameters that didn't matter, dead ends worth recording? These are lessons extracted from the run, not upfront design decisions._

| Strategy | Outcome | Why It Failed / What It Taught |
|---|---|---|
| _e.g. Remove load penalty, pure prefix affinity_ | _2.4× overload on one instance_ | _Hot-spotting; routing needs load awareness even when cache affinity is the goal_ |
| _e.g. Per-SLO hardcoded routing table_ | _+12% on one workload, −40% on another_ | _Overfits to one workload shape; dynamic approaches generalize better_ |
| ... | ... | ... |

_Include failed hypotheses, parameter ranges that were explored and ruled out, and architectural approaches that were tested and discarded. Even "we tried X and it made no difference" is valuable._

## 7. Cost

| Resource | Amount |
|---|---|
| Compute-hours | _e.g. 9 GPUs x 5.5 hrs = 49.5 GPU-hours_ |
| API / cloud cost | _e.g. ~$28 (Claude API) / $0 (self-hosted)_ |
| Wall-clock time | _e.g. 5.5 hours_ |
| Human effort | _e.g. 2 days calibration + 1 day run_ |

## 8. Artifacts

_Everything needed to reproduce or review. Mark each artifact's status so readers know what's committed vs. what needs to be regenerated._

| Artifact | Location | Status |
|---|---|---|
| Experiment config | `path/to/config` | _Committed_ |
| Raw results | `path/to/results/` | _Untracked — generated at runtime by `<command>`_ |
| Best output / code | `path/to/output` | _Committed / Untracked — see Appendix C_ |
| Deployment scripts | `path/to/scripts` | _Committed_ |
| Input data | `path/to/data` | _Committed / External_ |

## 9. Next Steps

_What would you do differently? What follow-up experiments does this suggest?_

1. Next step 1
2. Next step 2

---

## Appendix (optional)

### A. Scoring Formula

_Full mathematical definition of the scoring or evaluation function, if applicable._

### B. Configuration Files

_Embed the full content of every config file used to run the experiment. This makes the report self-contained — a reader with only this file has everything needed to reproduce the setup._

```yaml
# <filename>.yaml
# paste full content here
```

```yaml
# <filename2>.yaml
# paste full content here
```

### C. Best Output / Code

_The actual best result — code, configuration, model weights, etc. Include annotations explaining **why** it works._

### D. Raw Data Tables

_Generation-by-generation scores, metric timeseries, detailed measurements._

### E. Manual Reproduction Commands

_Verified step-by-step commands to reproduce the key result without the orchestration framework. Extract exact CLI flags from the run harness (e.g. `simulator.py`, `Makefile`) — do not infer._

```bash
# Step 1: build
# e.g. go build -o blis . (from inference-sim/)

# Step 2: run
# e.g. ./blis run --flag1 val1 --flag2 val2 ...
# Source: <harness_file>:<line_numbers>
```

_Expected output: <metric> ≈ <value>_

### F. Exploration Tree (optional)

_If your framework logs candidate lineage (e.g., OpenEvolve program database, population snapshots), include the full exploration history here. For sequential frameworks (OpenGlia, ShinkaEvolve), the generation-by-generation table in Section 5.2 or Appendix D typically suffices._

| ID | Parent | Score | Δ vs Parent | Status | What Changed (1-line) |
|---|---|---|---|---|---|
| seed | — | _0.00_ | — | baseline | _Initial program_ |
| _i9_ | _seed_ | _6.99_ | _+6.99_ | _★ new best_ | _Added session-aware prefix weights_ |
| _i10_ | _i9_ | _−100K_ | — | _✗ build fail_ | _Invalid Go syntax in weight calc_ |
| ... | ... | ... | ... | ... | ... |
