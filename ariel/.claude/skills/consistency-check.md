# consistency-check skill

Check that $ARGUMENTS is described consistently across all four layers of the project.

Run three parallel Explore agents:

**Agent 1 (Code)**: Search `selection_ellipse.py`, `ellipse_sweep.py`, `select_v2.py`, `set_fiducial.py`, `tophat.stan`, `normal.stan`, `predict.py`, `predict_cov.py` for all uses of $ARGUMENTS. Report exact values, variable names, and file:line locations.

**Agent 2 (Docs)**: Search `doc/model1.tex`, `doc/model2.tex`, `doc/model3.tex`, `Selection_v2.md`, `TFFit.md`, `Predict.md`, `DR1.md`, `README.md` for all references to $ARGUMENTS. Report exact wording, equations, and file locations.

**Agent 3 (Paper)**: Use `mcp__latex-server__get_latex_structure` then search `paper/main.tex` for all references to $ARGUMENTS. Report exact wording, equations, and section locations.

After all three agents complete, compare their findings and report:
1. Any discrepancies between layers (value differences, missing descriptions, contradictory equations)
2. Which layer is authoritative (code is truth)
3. Specific edits needed to bring docs and paper into alignment with code
