---
name: roux-change
description: Make, fix, implement or carry out a change (a TODO item) in the roux repository, from any repository (roux defaults to ../roux). Writes the item's backlog entry and phase plan (kept outside the repository), plans it in phases, verifies each phase (ruff, import check, the example-notebook tests) and stops for the user to review and commit. Can run inline or as a background sub-agent.
argument-hint: "<change or TODO id, e.g. H3> [path to roux, default ../roux]"
---

# roux-change

Carry out one change in the roux repository, using the TODO -> plan -> phases -> review/commit routine.
Arguments: `$ARGUMENTS`
- **Change**: a free-text change, or the ID of an item in the backlog (e.g. `H3`, `M1`). The backlog is kept
  outside the repository (the user's `strategy_roux_workflow` artifact, section "Repository backlog"). For an ID,
  use the entry text the user gives; if none is given, ask for it. Never create `.claude/TODOs.md`.
- **Roux path** (optional, last argument if it is a directory): the path to the roux repository, relative
  to the current directory. The default is `../roux`. Below, `ROUX` stands for this path.

All commands run from the current directory and point at `ROUX` (`git -C ROUX`, `--roux ROUX`), so
they work from any repository. Read `ROUX/.claude/CLAUDE.md` first for the commands, the architecture and the style.

## Driver

`ROUX/.claude/skills/roux-change/driver.py` is stdlib-only and runs with any `python3`. Its checks use `ROUX/.venv`.

```bash
python3 ROUX/.claude/skills/roux-change/driver.py info   --roux ROUX   # path, branch, HEAD, status
python3 ROUX/.claude/skills/roux-change/driver.py verify --roux ROUX   # checks on the uncommitted change; exit 0 ok, 1 failed, 2 environmental
python3 ROUX/.claude/skills/roux-change/driver.py verify --roux ROUX --tests tests/test_examples.py::test_roux_lib_df
python3 ROUX/.claude/skills/roux-change/driver.py verify --roux ROUX --no-tests
```

`verify` works on the files changed vs HEAD (tracked and untracked) under `roux/` and `tests/`:
- `ruff check --isolated --select F821,F811,F823` (undefined names and redefinitions) and `ruff format --check`.
- An import of every `roux.*` module. A failure blocks only if it is in a changed module. Others are
  listed as `known`, e.g. `roux.workflow.utils` until its H1 fix lands.
- pytest. By default it runs the tests that cover the changed modules (`roux/<a>/<b>.py` ->
  `tests/test_examples.py::test_roux_<a>_<b>`, plus `tests/test_jupytext_input.py` for
  `roux/workflow/io.py`, plus any changed test file). If it prints `tests: none cover...`, check the change
  by importing it and calling it. roux is installed editable in `ROUX/.venv`, so this works from any directory:
  `ROUX/.venv/bin/python -c 'from roux.lib.set import list2ranges; print(list2ranges([1,2,3]))'`.

## Routine

1. **State.** Run `driver.py info`. If files the change will touch already have uncommitted edits, stop
   and ask the user whether to continue on top of them.
2. **Backlog entry.** Write it in the report, not in a file (the user moves it into the backlog):
   - If the change is an existing item, use the entry the user gave.
   - If not, draft a `### <H|M|L><n>. <title>` entry (ask the user for the next free number). Rank it **High** if it
     is an error or an unexpected result in a public function, **Moderate** for robustness, tests, dependencies
     or security, and **Low** for hygiene. Include `file:line` locations and the evidence, in the style of
     the existing entries.
   - Never plan to remove "unused" functions or imports: roux is imported as a library.
3. **Plan.** Under the entry, write a `Plan:` list with one `- [ ] Phase N: <scope>; verify: <how>` line per
   phase. Use one phase if the change is small. Split it only where a reviewer would want separate
   commits, e.g. one phase per subpackage, or tests before a behaviour change. Each phase must leave the
   repository working.
4. **Phase loop.** For each phase:
   1. Make the edits, following roux's style (local imports, `##` comments, docstrings with `Parameters:`).
      Keep public signatures, and do not change unrelated behaviour.
   2. Run `driver.py verify --roux ROUX` until it exits 0. Add `--tests` for anything else the phase affects.
   3. Tick the phase (`- [x]`) in the entry.
   4. **Stop for review.** Report the phase, the current entry with its plan (a Markdown block to paste
      into the backlog), `git -C ROUX diff --stat`, the verify result, and a proposed
      commit message. Ask the user to review and commit. Do not start the next phase until they reply.
      Commit only if the user asks you to (`git -C ROUX commit`, ending the message with the attribution
      lines of this session), and never push.
5. **Done.** When every phase is ticked, append `Status: done` to the entry in the final report. Nothing
   about the backlog is committed to the repository.

## Running as a background sub-agent

A background sub-agent can't ask the user anything. So it runs the routine and **ends its turn at every
"Stop for review"** with the report as its final message. The parent relays the report to the user. After
the user commits, the parent resumes the same agent with SendMessage (e.g. "phase 1 committed,
continue"). Typical launch from the parent:

```
Agent(description="roux change H3", run_in_background=true,
      prompt="Follow ROUX/.claude/skills/roux-change/SKILL.md with: H3, roux path ../roux. Stop after each phase and report.")
```

The sub-agent also stops and reports, leaving the decision to the user, if a tool call is
denied (it can't answer permission prompts) or if `verify` exits 2.

## Good to know

- `tests/test_examples.py` appends to the IPython profile at import. `verify` points `IPYTHONDIR` at a
  temp dir, so run pytest through the driver, not directly.
- Tests that call `to_scr` leave `ROUX/.to_src/`. `verify` removes it if it was not there before.
- Example tests write `examples/outputs/` (git-ignored). This is expected.
- 6 example tests (`roux_lib_df`, `roux_lib_df_plot`, `roux_lib_io`, `roux_viz_annot`, `roux_viz_dist`,
  `roux_viz_scatter`) download data. Without network access (e.g. behind a restrictive proxy:
  `Tunnel connection failed: 403`) `verify` exits 2. That is unrelated to the change: report it and ask the user.
- `ruff` must be the pinned 0.15.10 in `ROUX/.venv`. A newer one strips `# noqa`.
- An example-notebook test checks that the notebook runs. For a bug fix, also call the fixed code
  path directly: it is often a branch the examples don't reach.

## Troubleshooting

- `not the roux repository (no pyproject.toml with name = "roux")`: wrong `--roux`. It is relative to
  the current directory, not to the skill.
- `missing ROUX/.venv/bin/ruff`: run `uv sync --all-extras` in `ROUX`.
- `FAIL: ruff format --check`: run `ROUX/.venv/bin/ruff format <file>` on the files listed.
