# RESEARCH ASSEMBLY LINE PROTOCOL v2.1 (2026)

**Target:** AI Research Codebase (Model Architectures & Visualization Tools)
**Owner:** [Your Name]
**Last Updated:** 2026-03-04

---

## 0. HOW TO USE THIS PROTOCOL

This file governs the two-agent workflow for developing and refining this codebase.
It is read by both AI agents and the human operator. Every prompt to either agent
should begin with:

> "Read protocol.md in the repo root before proceeding."

**The human operator is the final authority.** No agent may override the operator's
explicit instruction, even if it conflicts with a guideline below.

---

## 1. AGENT ROLES

### GEMINI (The Architect / Auditor)

* **Strengths:** 1M+ token context, full-repo reasoning, cross-file dependency tracking.
* **Primary Tasks:** Full-repo audits, managing `ROADMAP.md`, creating `tasks/active/` files, large-scale refactors.
* **Secondary Tasks:** Generating test scaffolds, profiling, identifying dead code.

### CLAUDE (The Surgeon / Designer)

* **Strengths:** High-fidelity math reasoning, precise refactoring, UI/UX (React).
* **Primary Tasks:** Component implementation, einsum correctness, kernel logic, visualization tools, executing `tasks/active/` files.
* **Secondary Tasks:** Docstring authoring, mathematical proofing.

### When to use which

* "I need to understand how module X affects the rest of the repo" → **Gemini**
* "I need to rewrite this einsum contraction and prove it's equivalent" → **Claude**
* "I need a profiling summary across 20+ files" → **Gemini**
* "I need a React dashboard for attention heatmaps" → **Claude**
* **Ambiguous?** → Default to Gemini for analysis, Claude for implementation.

---

## 2. PROJECT MANAGEMENT & STATE

**Objective:** Maintain persistent state outside the LLM context window using Git and Markdown.

### 2.1 The Two-Level State System

1. **STRATEGY (`ROADMAP.md`):**
    * **Scope:** High-level milestones (e.g., "Phase 1: Base Kernel", "Phase 2: Viz").
    * **Owner:** **Gemini** & Operator.
    * **Usage:** Read during planning and audits.

2. **EXECUTION (`tasks/active/task-ID-name.md`):**
    * **Scope:** Detailed instructions for the *current* git branch only.
    * **Owner:** **Claude**.
    * **Usage:** Read continuously while coding.
    * **Lifecycle:** Created by Gemini → Worked on by Claude → Moved to `tasks/archive/` upon PR merge.

### 2.2 Task File Template

Every file in `tasks/active/` must follow this structure:

````markdown
# Task-00X: [Short Title]

**Branch:** `feat/task-name`
**Status:** `TODO | IN_PROGRESS | REVIEW | DONE`
**Created by:** GEMINI | OPERATOR
**Assigned to:** CLAUDE

## Objective
One paragraph describing what this task accomplishes and why.

## Scope (files allowed to modify)
- `src/kernels/conv.py`
- `tests/test_conv.py`

## Acceptance Criteria
- [ ] FFT convolution replaces naive loop in `forward()`
- [ ] All existing tests pass
- [ ] New test covers edge case for non-square K

## Relevant Invariants
- INV-1: Tensor shapes match documented shapes
- INV-2: Initialization scheme preserved

## Context / References
- Related to ROADMAP Phase 1, Milestone 2
- See `docs/architecture.md` for design notes
````

### 2.3 Git Hygiene

* **Main Branch:** `main` (Protected. Never push directly.)
* **Task Branches:** `feat/task-name` (Features), `fix/issue-name` (Bugs), `exp/experiment-name` (Research).
* **Commit Protocol:** Agents must suggest clear commit messages referencing the Task ID (e.g., `feat: implement FFT conv (ref #Task-002)`).

---

## 3. TECHNICAL CONSTRAINTS

### 3.1 Invariants (NEVER violate without operator approval)

These are correctness properties. Breaking any of these silently is a critical failure.

| ID    | Invariant                                                                          |
|-------|------------------------------------------------------------------------------------|
| INV-1 | Documented tensor shapes must match actual runtime shapes                          |
| INV-2 | Model-specific initialization schemes must not be altered without operator approval |
| INV-3 | Existing benchmarks must not regress (verify before and after)                     |
| INV-4 | No silent dependency additions (new imports must be noted in `RECAP_STATE`)        |
| INV-5 | Public API signatures do not change without operator approval                      |
| INV-6 | Architecture-specific constraints listed in the active task file must be honored    |

### 3.2 Preferences (follow unless justified in `RECAP_STATE`)

These are quality guidelines. Deviations are acceptable if the agent explains why.

| Pref   | Guideline                                                              |
|--------|------------------------------------------------------------------------|
| PREF-1 | Use `torch.einsum` for multi-dimensional contractions                  |
| PREF-2 | Document tensor shapes in comments: `# [B, C, L]`                     |
| PREF-3 | Visualization tools use scientific colormaps (viridis, magma, inferno) |
| PREF-4 | No generic CSS frameworks (Bootstrap, etc.) in research viz tools      |
| PREF-5 | One function = one clear responsibility                                |
| PREF-6 | Logging over print statements; use Python `logging` module             |

---

## 4. THE DIFF-FIRST RULE

Before writing or rewriting any code, the agent **must** first output a structured
change plan:

```markdown
## CHANGE PLAN
- **File:** src/kernels/conv.py
  - **Function:** `forward()` (line ~45)
  - **Action:** Replace naive loop with einsum contraction
  - **Reason:** O(N²) → O(N) for the D×K dimensions
  - **Risk:** Shape mismatch if K is not square — will add assert

- **File:** tests/test_conv.py
  - **Action:** Add shape-check test for new einsum path
```

The operator (or the next agent reading the handoff) can approve, modify, or reject
the plan before code is generated.

**Exception:** Trivial changes (typo fixes, comment updates) may skip the change plan.

---

## 5. THE HANDSHAKE — `RECAP_STATE`

Every agent handoff **must** include a `RECAP_STATE` block. This is the single
source of truth that travels between agents. Without it, the next agent operates blind.

### 5.1 Format (copy this template exactly)

```yaml
RECAP_STATE:
  git_context:
    branch: "feat/..."
    active_task_file: "tasks/active/task-00X.md"
  status: "<one-line summary of what was done>"
  files_touched:
    - path/to/file1.py  # brief note on change
    - path/to/file2.py  # brief note on change
  tensor_shapes:
    input: [B, C, L]
    output: [B, C, L]
  invariants_held:
    - "INV-1: Tensor shapes verified"
    - "INV-2: Initialization scheme preserved"
  invariants_broken:
    - "None"  # or: "INV-1: output shape changed from [B,C,L] to [B,C,L,2] — see open_questions"
  tests_passing: true | false | untested
  next_step:
    agent: "CLAUDE | GEMINI"
    instruction: "<specific next action>"
    scope:
      - path/to/allowed_file1.py
      - path/to/allowed_file2.py
  open_questions:
    - "<unresolved issues for operator>"
```

### 5.2 Rules

1. The `RECAP_STATE` block is placed at the **end** of every agent's output.
2. The human operator pastes it at the **top** of the prompt to the next agent.
3. If `invariants_broken` contains anything other than `"None"`, the operator **must** acknowledge before proceeding.
4. If `tests_passing` is `false` or `untested`, the next agent's **first action** must be to diagnose or run tests.
5. `open_questions` halts the pipeline — the operator resolves them before the next handoff.

---

## 6. VALIDATION BETWEEN HANDOFFS

### 6.1 Operator smoke test

After every agent pass and before the next handoff, the operator should run:

```bash
# Adjust to your repo
python -m pytest tests/ -x --tb=short    # stop on first failure
python -c "from src import kernels"       # import check
```

### 6.2 Agent-side validation

Each agent should, when possible:

1. Include a runnable snippet or test that validates its changes.
2. Predict the output shape / expected behavior before the code runs.
3. Flag if it **cannot** validate (e.g., needs GPU, needs dataset) in `open_questions`.

### 6.3 Validation status in `RECAP_STATE`

* `tests_passing: true` — agent ran or reasoned through tests, all green.
* `tests_passing: false` — something broke, details in `open_questions`.
* `tests_passing: untested` — agent could not verify (explain why).

---

## 7. ESCALATION — THE `QUESTION` BLOCK

When an agent encounters ambiguity, uncertainty, or a decision that depends on
research intent rather than code correctness, it must **halt and ask** rather than guess.

```markdown
## QUESTION
- **Context:** [what the agent was trying to do]
- **Ambiguity:** [what is unclear]
- **Options:**
  1. [Option A] — tradeoff: ...
  2. [Option B] — tradeoff: ...
- **Agent's lean:** [which option and why]
- **Blocking:** [yes | no — can the agent continue on other parts while waiting?]
```

**Rules:**
* A `QUESTION` block means the operator must respond before that specific subtask continues.
* If `Blocking: no`, the agent may continue with unrelated parts of the task.
* Never fabricate an answer to avoid asking a question.

---

## 8. CONFLICT RESOLUTION — THE CRITIC LOOP

If Agent B wants to revert or significantly alter Agent A's work, it **may not**
do so without completing the Critic Loop.

```markdown
## CRITIC LOOP
- **Original change by:** [Agent A]
- **What was done:** [summary]
- **Original rationale:** [why Agent A made this choice]
- **Proposed revert/change:** [what Agent B wants to do instead]
- **Justification type:** [correctness bug | performance regression | readability | architectural conflict]
- **Evidence:** [benchmark numbers, shape mismatch proof, or concrete code example]
- **Recommendation:** [revert | partial revert | keep original + patch]
```

**Rules:**
1. Justification must be **measurable or provable** — not "this seems cleaner."
2. `performance regression` requires numbers or complexity analysis.
3. `correctness bug` requires a failing case or shape proof.
4. The operator makes the final call on all Critic Loop outcomes.

---

## 9. PROMPT TEMPLATES

### 9.1 Architect Pass (Gemini)

```
Read protocol.md before proceeding.

<paste RECAP_STATE here, or "First pass — no prior state.">

ROLE: Architect/Auditor. Your job is analysis and planning, not implementation.

TASK: [describe the audit, profiling, or planning task]

SCOPE: [list of files/directories to examine]

OUTPUT REQUIREMENTS:
1. A CHANGE PLAN (Section 4) if recommending modifications.
2. A new or updated task file (Section 2.2) if creating work for Claude.
3. A RECAP_STATE block (Section 5) at the end of your response.
4. Any QUESTION blocks (Section 7) if you hit ambiguity.

CONSTRAINTS: Follow all invariants in Section 3.1. Note any preference deviations.
```

### 9.2 Surgical Pass (Claude)

```
Read protocol.md before proceeding.

<paste RECAP_STATE here>

ROLE: Surgeon/Designer. Your job is precise implementation and UI work.

TASK: [describe the specific implementation or UI task, or reference task file]

ACTIVE TASK FILE: tasks/active/task-00X.md

OUTPUT REQUIREMENTS:
1. A CHANGE PLAN first (Section 4), then the implementation.
2. Inline shape comments on all tensor operations (PREF-2).
3. A runnable validation snippet or test if possible (Section 6.2).
4. A RECAP_STATE block (Section 5) at the end of your response.
5. Any QUESTION or CRITIC LOOP blocks as needed (Sections 7–8).

CONSTRAINTS: Follow all invariants in Section 3.1. Stay within task file scope.
```

---

## 10. WORKFLOW SUMMARY

```
┌──────────────────────────────────────────────────────────────┐
│  OPERATOR defines task + scope                               │
│  ↓                                                           │
│  GEMINI (Architect Pass)                                     │
│    → Reads ROADMAP.md, analyzes repo                         │
│    → Creates/updates tasks/active/task-00X.md                │
│    → Outputs CHANGE PLAN + RECAP_STATE                       │
│  ↓                                                           │
│  OPERATOR reviews plan, resolves QUESTIONs                   │
│  ↓                                                           │
│  CLAUDE (Surgical Pass)                                      │
│    → Reads task file, implements per CHANGE PLAN             │
│    → Outputs code + tests + RECAP_STATE                      │
│  ↓                                                           │
│  OPERATOR runs smoke tests (Section 6)                       │
│  ↓                                                           │
│  If issues → CRITIC LOOP or new task iteration               │
│  If clean  → PR, merge, move task to tasks/archive/          │
└──────────────────────────────────────────────────────────────┘
```

---

## CHANGELOG

| Version | Date       | Changes                                                                                                    |
|---------|------------|------------------------------------------------------------------------------------------------------------|
| 1.0     | —          | Initial draft                                                                                              |
| 1.2     | —          | Added Critic Loop concept                                                                                  |
| 2.0     | 2026-03-04 | Structured RECAP_STATE, Diff-First rule, Critic Loop spec, QUESTION escalation, prompt templates, validation steps, invariant/preference split, file ownership |
| 2.1     | 2026-03-04 | Two-level state system (ROADMAP + task files), task file template with scope and acceptance criteria, git hygiene, restored `invariants_broken` and `scope` fields in RECAP_STATE, deduplicated heuristics, streamlined section ordering |