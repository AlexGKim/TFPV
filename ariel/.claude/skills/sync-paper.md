# sync-paper skill

Update `paper/main.tex` to match the current code and doc/ notes for $ARGUMENTS.

Steps:
1. Read the relevant code sections (use CLAUDE.md parameter registry and component map to identify which files)
2. Read the corresponding `doc/*.tex` notes for formal math
3. Use `mcp__latex-server__get_latex_structure` to locate the relevant section(s) in `paper/main.tex`
4. Read those sections of `paper/main.tex`
5. Edit `paper/main.tex` using `mcp__latex-server__edit_latex_file` to match the code/doc truth
6. Run `mcp__latex-server__validate_latex` to confirm no LaTeX errors introduced
7. Report what changed and why
