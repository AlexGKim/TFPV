# TFPV/ariel — Claude Code Instructions

## Component Map (Authority Hierarchy)

| Concept | Authoritative Source | Narrative | Implementation | Publication |
|---------|---------------------|-----------|----------------|-------------|
| Selection algorithm | `doc/model1.tex` | `Selection_v2.md` | `selection_ellipse.py`, `ellipse_sweep.py` | `paper/main.tex` |
| TFR fitting model | `doc/model2.tex`, `doc/model3.tex` | `TFFit.md` | `tophat.stan`, `normal.stan` | `paper/main.tex` |
| Prediction step | `doc/model2.tex` | `Predict.md` | `predict.py`, `predict_cov.py` | `paper/main.tex` |
| DR1 run commands | `DR1.md` | — | — | — |

**Formal math is truth.** When code and `doc/*.tex` disagree, the math wins. Update the code to match.

---

## Key Parameter Registry

These are ground-truth values derived from `doc/*.tex`. File:line pointers show where each value is implemented in code — if they diverge from the math, fix the code.

| Parameter | Value | Authoritative Location(s) |
|-----------|-------|--------------------------|
| `y_min` offset below `haty_min` | −0.5 mag | `selection_ellipse.py:507`, `ellipse_sweep.py:310` |
| `y_max` offset above `haty_max` | +1.0 mag | `selection_ellipse.py:508`, `ellipse_sweep.py:311` |
| `z_obs_min` default | 0.03 | `select_v2.py:223` (CLI default) |
| `z_obs_max` default | 0.1 | `set_fiducial.py:78` (CLI default) |
| `haty_min` loose pre-filter default | −23.0 | `selection_ellipse.py:729` |
| `haty_max` loose pre-filter default | −18.0 | `selection_ellipse.py:731` |

**Distinction**: `haty_min`/`haty_max` are the ellipse-derived selection boundaries.
`y_min`/`y_max` are the Stan model integration bounds, set wider by the offsets above.

---

## Consistency Rule

**When changing any parameter value, algorithm step, or equation in `doc/*.tex`:**

1. Update the corresponding code (`.py`, `.stan`) to implement the new math
2. Search `Selection_v2.md`, `TFFit.md`, `Predict.md`, `DR1.md` for references — update narrative
3. Search `paper/main.tex` for references — update publication text
4. Use `mcp__latex-server__validate_latex` after every edit to `paper/main.tex` or `doc/*.tex`

For broad consistency sweeps, use the `/consistency-check` skill.

---

## MCP Tools — Use Systematically

- `mcp__latex-server__get_latex_structure` — run **before** editing any `.tex` file to locate the right section
- `mcp__latex-server__validate_latex` — run **after** every `.tex` edit
