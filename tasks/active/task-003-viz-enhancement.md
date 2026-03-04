# Task-003: Visualization Redesign & Gradio Enhancements (Milestone 1.2)

**Branch:** `feat/viz-clarity-improvements`
**Status:** `COMPLETED`
**Created by:** Operator / GEMINI
**Assigned to:** GEMINI (Architect Pass) -> CLAUDE (Surgical Pass)

## Objective
Execute Milestone 1.2 from the ROADMAP: Redesign panels, add interpretive context, improve the spatial layout of the Gradio app, and introduce new capabilities like cross-head comparisons.

## Scope (files allowed to modify)
- `scripts/visualize_hyena_kernels_app.py`
- `scripts/visualize_hyena_kernels.py`

## Acceptance Criteria
- [x] **Interpretive Context:** Add Markdown summaries, reference ranges, and annotations directly into the Gradio UI so users understand *how* to read each panel without checking the source code.
- [x] **Gradio Layout & Consistency:** Improve the spatial layout of the Gradio app. Ensure consistent control placement across the "Global Kernels", "Feature Mixing", and "Activations" tabs.
- [x] **Cross-Head Comparison:** Add a feature to the Kernel Structure tab to allow side-by-side or aggregated comparisons across multiple heads simultaneously.
- [x] **Alternative Visualizations:** Replaced dense Mixing Matrices panel with SVD σ₁ heatmap.

## Relevant Invariants
- INV-4: No new dependencies without operator approval.
- INV-6: No changes to kernel extraction or model loading logic.
- PREF-1: Gradio app layout must be logical and use Blocks/Rows/Columns effectively.

## Next Steps (Architect Pass)
1. Read `scripts/visualize_hyena_kernels_app.py` to understand the current Gradio layout.
2. Propose a concrete `CHANGE PLAN` for the Gradio UI restructuring and the exact implementation of the cross-head comparison.
3. Update this task file with granular checklist items for the Surgical pass.
