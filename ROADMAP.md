# ROADMAP

**Project:** AI Research Codebase — Model Architecture & Visualization Tools
**Last Updated:** 2026-03-04

---

## Phase 1: Visualization Clarity & Best Practices

**Goal:** Make the existing kernel visualization tooling self-explanatory and
scientifically rigorous. A colleague unfamiliar with the model should be able
to open the Gradio app, explore the panels, and understand what each
visualization reveals about the model's behavior — without reading the source code.

### Milestone 1.1 — Audit & Improvement Plan
- [x] Full audit of all 11 panels: clarity, best practices, interpretive scaffolding
- [x] Identify panels that need redesign vs. panels that need minor polish
- [x] Research alternative/complementary visualization techniques
- [x] Produce prioritized task files for Claude

### Milestone 1.2 — Panel Redesign & Enhancement
- [x] Add interpretive context to each panel (reference ranges, annotations, summaries)
- [x] Introduce alternative visualizations where current ones are hard to read
- [x] Improve spatial layout and consistency across tabs in Gradio app
- [x] Add cross-head comparison capability to Kernel Structure tab

### Milestone 1.3 — UI/UX Polish
- [ ] Consistent tab ↔ slider mapping (clear which controls affect which tab)
- [ ] Loading states, error handling, edge cases
- [ ] Export / reporting workflow from Gradio app
- [ ] Documentation for new researchers

---

## Phase 2: [Future — Model Architecture Work]

*To be defined.*