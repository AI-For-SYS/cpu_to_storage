---
name: experiment
description: >
  Help write experiment reports for compute experiments
  (inference, training, simulation, benchmarking, optimization).
  Use when the user mentions experiment report, writing results,
  benchmark report, documenting an experiment, experiment writeup,
  filling out the experiment template, or wants to document what
  they ran and what happened — even if they don't say "experiment"
  explicitly.
argument-hint: "[e.g. 'write report for my benchmark run' or 'document the calibration results']"
---

**Skill creator & maintainer**: Yevgeny Burshtein — for questions, bugs, or feature requests about this skill, contact Yevgeny directly.

Help the user write a structured experiment report using the template in this skill directory.

When responding:
- Ask questions to understand what was run — don't assume hardware, software, or tools
- Be concrete: fill in tables, compute deltas, format numbers
- Work section by section — confirm each before moving on
- Look for data sources in the project (logs, CSV, SQLite, API output, config files) and offer to pull metrics from them
- The report should be self-contained: someone else should be able to read it and understand what happened, why, and how to reproduce it

## What the user asked

$ARGUMENTS

## Workflow

1. **Understand the experiment** — ask:
   - What did you run? (model, simulation, benchmark, optimization...)
   - What compute? (GPUs, CPUs, cluster, cloud, local machine...)
   - What were you trying to learn?
   - Do you have results data I can look at? (files, logs, metrics)

2. **Read the template** — open [report-template.md](report-template.md) in this directory

3. **Fill sections interactively** — for each section (0–9):
   - Ask for the information needed
   - Fill the section with concrete values
   - Show the draft to the user
   - Move on when confirmed

4. **Reproducibility gate** — before writing the report, verify these mandatory fields are present. If any are missing, **stop and ask the user** — do not write the report with placeholders.

   **MUST HAVE (blocks report):**
   - [ ] Repo URL + branch + commit SHA for the orchestration framework (Section 3.2)
   - [ ] Repo URL + commit SHA for every external dependency (Section 3.2)
   - [ ] All config files embedded verbatim in Appendix B (not just referenced)
   - [ ] Baseline measurement with at least one metric (Section 3.5)
   - [ ] At least one evolution result (Section 5.1)
   - [ ] Reproduction commands verified against the actual run harness (Appendix E)
   - [ ] Search space defined — what's mutable vs fixed (Section 0.2)

   **SHOULD HAVE (warn but allow):**
   - [ ] Dependency lock file location (Section 4.2)
   - [ ] `git status` run before Section 8
   - [ ] Cost breakdown (Section 7)
   - [ ] Exploration insights (Section 6.4)

   The standard is: **can someone reproduce this experiment from the report alone?** If not, the report isn't ready.

5. **Output the report** — write the completed report as a markdown file to `experiments/`. Suggest a filename like `experiments/<framework>_experiment_report_<short_name>.md` or let the user choose.

6. **Review** — read back the full report, check for:
   - Missing data (empty cells, TBD placeholders)
   - Consistency (numbers in Results match numbers in Setup)
   - Reproducibility gate passed (step 4)

7. **Update README** — read `README.md` at the repo root. Add a row to the appropriate framework table in the Experiment Index section with: report link, author, target, and key result. If the framework doesn't have a table yet, create one following the existing pattern.

8. **Generate Slack summary** — draft a short, copy-paste-ready message the user can share. Pull from:
   - Title from report header
   - Framework + model from Section 3.2
   - Target from Section 0.1
   - Result: best metric vs baseline with % change, wall-clock time, and cost from Sections 5.1 and 7
   - Key insight: first finding from Section 6.1 (one sentence)
   - Link: `https://github.ibm.com/AI4SYS/evolve_experiments/blob/main/<report_path>`

   Format:
   ```
   📊 New experiment report: <title>

   Framework: <framework> (<mode>, <model>)
   Target: <evolution target — one line>
   Result: <best metric change> in <wall-clock time>, <cost>
   Key insight: <one sentence from 6.1>

   Report: <github link>
   ```

   Show this to the user at the end of the conversation so they can copy-paste it to Slack.

## Section-by-section guidance

### 0. Design
- This section captures what was planned **before** the experiment ran. Ask:
  - **Evolution target (0.1)**: What function/algorithm/code is being evolved? What are its inputs and outputs?
  - **Search space (0.2)**: What's mutable vs fixed? Use a table. Include code structure, parameters, weights — anything the framework can change. Also list what's explicitly held constant.
  - **Constraints (0.3)**: Runtime limits, language restrictions, available libraries, timeout budgets, interface contracts the evolved code must satisfy.
  - **Success criteria (0.4)**: What quantitative target was set before seeing results? This is different from the Hypothesis — it's the bar for "did this work?"
- If the user doesn't have a formal design phase, help reconstruct it from what they describe. Frame it as "what would you have written down before hitting run?"
- Keep it short — one paragraph for target, one table for search space, one paragraph each for constraints and criteria.

### 1. Objective
- One paragraph max
- Should answer: "What question does this experiment answer?"
- Bad: "We ran some benchmarks" — Good: "Determine whether X outperforms Y under condition Z"

### 2. Hypothesis
- What did the experimenter expect, and why?
- If they don't have one, help formulate it from their description

### 3. Setup
- **Hardware**: Ask what compute was used. Don't assume GPUs — could be CPUs, FPGAs, TPUs, or a laptop.
- **Software**: Ask about versions. Include model name/size if applicable. For any orchestration framework or custom tool, capture the repo URL, branch, and commit SHA — this is **required**. Check git remotes (`git remote -v`), README files, and config files first. If the repo URL is not found anywhere, **ask the user explicitly — do not leave it blank or use a placeholder**.
- **External dependencies**: Are there any external tools or simulators cloned separately (not part of the main repo)? If so, capture the repo URL, commit SHA, and where they must be placed. **REQUIRED: run `git remote -v` inside the dependency's directory** to get the exact URL — never infer or guess it from context, README files, or the directory name. If the directory exists but has no git remote, ask the user explicitly.
- **Configuration**: Every tunable parameter someone would need to reproduce the experiment. Use tables.
- **Workload**: What input data or traffic pattern? How generated? Why this workload?
- **Baseline**: What is the control condition? How was it measured?

### 4. Methodology
- Step-by-step, enough for someone else to re-run
- Include a reproducibility checklist (configs committed? seeds fixed? data available?)
- **Manual reproduction commands**: if the experiment uses a run harness (e.g. `simulator.py`, a `Makefile`, a shell wrapper), **read that file** to extract the exact CLI flags passed to the underlying tool. Never infer flags from documentation or memory — use what the harness actually passes. Record these verified commands in Appendix E.
- **Do NOT reference untracked paths in reproduction commands.** Commands in Appendix E must only reference files committed to the repo. Never `cp` from untracked output directories (e.g. snapshot dirs). If the best output artifact is not committed, instruct the reader to recreate it from Appendix C (the embedded best output) — e.g. "copy the code from Appendix C into `<path>`".

### 5. Results
- **Summary table**: baseline vs best, with deltas and % change
- **Detailed results**: distributions, charts, per-component breakdowns
- **Failure modes**: what went wrong, how often, why — quantify with a table

### 6. Analysis
- Did the hypothesis hold?
- Key findings (numbered list)
- Limitations — what can't you conclude?
- Comparison to prior experiments if any
- **Exploration insights (6.4)**: Ask what the search process revealed — strategies that failed, parameters that didn't matter, dead ends worth recording. These are lessons extracted from the run, not upfront decisions. Capture the strategy, what happened, and what it taught. This prevents future experiments from re-exploring known dead ends. Even "we tried X and it made no difference" is worth recording. If there are no notable insights, skip the subsection.

### 7. Cost
- Compute-hours (e.g., "9 GPUs x 5.5 hrs = 49.5 GPU-hours")
- API/cloud costs if applicable
- Wall-clock time
- Human effort (calibration, monitoring, debugging)

### 8. Artifacts
- 3-column table: Artifact, Location, Status
- Use relative paths where possible
- **Run `git status` before writing this section.** Mark each artifact as Committed, Untracked, or External
- List both committed and untracked artifacts — readers need to know what exists and what they'd need to regenerate
- For untracked runtime output, note the command that generates it

### Appendix B. Configuration Files
- Embed the **full content** of every config/YAML file used to run the experiment
- Look for config files in the project and read them — don't ask the user to paste them manually
- Goal: the report must be self-contained; a reader with only this markdown file can reconstruct the exact setup

### 9. Next Steps
- What would you do differently?
- What follow-up experiments does this suggest?
- Keep actionable — not vague wishes

## Reference files

- [report-template.md](report-template.md) — the 9-section markdown template to fill out
- [examples/](examples/) — completed reports showing the template in use
