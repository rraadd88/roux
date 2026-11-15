#!/usr/bin/env python3
"""Driver of the roux-change skill: locate the roux repository, show its state, verify a change.

Runs from any directory with any python3 (stdlib only); the checks use the tools of `<roux>/.venv`.

Usage:
    python3 <roux>/.claude/skills/roux-change/driver.py info   [--roux ../roux]
    python3 <roux>/.claude/skills/roux-change/driver.py verify [--roux ../roux] [--tests NODEID ...] [--no-tests]

`verify` exits 0 if every blocking check passes, 1 if a check failed, 2 if a check could not
run for an environmental reason (e.g. no network): then stop and ask the user.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

## blocking ruff codes: undefined names, redefinitions (F401 stays ignored: roux re-exports names)
RUFF_SELECT = "F821,F811,F823"
## output of failures that lie outside the repository
ENV_ERRORS = (
    "Tunnel connection failed",
    "Temporary failure in name resolution",
    "Network is unreachable",
    "PermissionError",
    "Read-only file system",
)


def get_roux(path: str) -> Path:
    """Resolve and check the roux repository path."""
    p = Path(path).resolve()
    pyproject = p / "pyproject.toml"
    if not pyproject.is_file() or not re.search(
        r'^name\s*=\s*"roux"', pyproject.read_text(), flags=re.M
    ):
        sys.exit(f"not the roux repository (no pyproject.toml with name = \"roux\"): {p}")
    for tool in ["python", "ruff"]:
        if not (p / ".venv/bin" / tool).exists():
            sys.exit(f"missing {p}/.venv/bin/{tool}: run `uv sync --all-extras` in {p}")
    return p


def run(cmd: list, roux: Path, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=roux, env=env, capture_output=True, text=True)


def git(roux: Path, *args) -> str:
    return run(["git", *args], roux).stdout.strip()


def changed_files(roux: Path) -> list:
    """Tracked files changed vs HEAD, plus untracked files (git-ignored ones excluded)."""
    files = git(roux, "diff", "--name-only", "HEAD").splitlines()
    files += git(roux, "ls-files", "--others", "--exclude-standard").splitlines()
    ## untracked dotfiles/dirs at the root (shell profiles, IDE settings) are not part of a change
    keep = lambda f: not f.startswith(".") or f.startswith((".claude/", ".github/"))
    return sorted({f for f in files if f and keep(f)})


def info(roux: Path) -> int:
    print(f"roux:   {roux}")
    print(f"branch: {git(roux, 'branch', '--show-current')}")
    print(f"HEAD:   {git(roux, 'log', '--oneline', '-1')}")
    print("status:")
    print("\n".join("  " + s for s in git(roux, "status", "--short").splitlines()) or "  clean")
    return 0


def tests_for(files: list, roux: Path) -> list:
    """pytest node ids covering the changed modules: roux/<a>/<b>.py -> test_roux_<a>_<b>."""
    defined = set(
        re.findall(r"^def (test_\w+)", (roux / "tests/test_examples.py").read_text(), flags=re.M)
    )
    ids = []
    for f in files:
        p = Path(f)
        if p.parts[0] == "tests" and p.suffix == ".py":
            ids.append(f)
            continue
        if p.parts[0] != "roux" or p.suffix != ".py":
            continue
        name = "test_" + "_".join(p.with_suffix("").parts).replace("___init__", "")
        if name in defined:
            ids.append(f"tests/test_examples.py::{name}")
        if f == "roux/workflow/io.py":
            ids.append("tests/test_jupytext_input.py")
    return sorted(set(ids))


def verify(roux: Path, tests: list, no_tests: bool) -> int:
    py, ruff = str(roux / ".venv/bin/python"), str(roux / ".venv/bin/ruff")
    files = changed_files(roux)
    pys = [
        f
        for f in files
        if f.startswith(("roux/", "tests/")) and f.endswith(".py") and (roux / f).exists()
    ]
    print("changed: " + (", ".join(files) or "nothing"))
    results = {}

    ## lint the changed python files
    if pys:
        r = run([ruff, "check", "--isolated", "--select", RUFF_SELECT, *pys], roux)
        results[f"ruff {RUFF_SELECT}"] = (r.returncode == 0, r.stdout + r.stderr)
        r = run([ruff, "format", "--check", *pys], roux)
        results["ruff format --check"] = (r.returncode == 0, r.stdout + r.stderr)

    ## every module of the package imports (a module that fails at HEAD is reported, not blocking)
    code = (
        "import importlib,pkgutil,roux\n"
        "for m in pkgutil.walk_packages(roux.__path__,'roux.'):\n"
        "    try: importlib.import_module(m.name)\n"
        "    except Exception as e: print(m.name, type(e).__name__, e, sep='\\t')\n"
    )
    r = run([py, "-c", code], roux)
    failed = [l.split("\t") for l in r.stdout.splitlines() if l.count("\t") >= 2]
    changed_mods = {f[:-3].replace("/", ".").removesuffix(".__init__") for f in pys}
    blocking = [l for l in failed if l[0] in changed_mods]
    for l in failed:
        print(f"import: {'FAIL' if l in blocking else 'known'} {' '.join(l)}")
    results["import every module"] = (not blocking and r.returncode == 0, r.stderr)

    ## tests (the IPython profile goes to a temporary dir: tests/test_examples.py appends to it)
    tests = tests or ([] if no_tests else tests_for(files, roux))
    if tests:
        print("tests: " + " ".join(tests))
        had_to_src = (roux / ".to_src").exists()
        with tempfile.TemporaryDirectory() as ipy:
            env = {**os.environ, "IPYTHONDIR": ipy}
            r = run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], roux, env=env)
        if not had_to_src and (roux / ".to_src").exists():
            import shutil

            shutil.rmtree(roux / ".to_src")  ## scratch dir of to_scr, left by the tests
        results["pytest"] = (r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:])
    elif not no_tests:
        print("tests: none cover the changed modules; check the change by importing and calling it")

    env_failure = False
    for name, (ok, out) in results.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            print("\n".join("  " + l for l in out.strip().splitlines()))
            env_failure |= any(e in out for e in ENV_ERRORS)
    if all(ok for ok, _ in results.values()):
        return 0
    if env_failure:
        print("ENV: a failure looks environmental (network/permission): stop and ask the user")
        return 2
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["info", "verify"])
    parser.add_argument("--roux", default="../roux", help="path to the roux repository")
    parser.add_argument("--tests", nargs="*", default=[], help="pytest node ids (default: inferred)")
    parser.add_argument("--no-tests", action="store_true", help="skip pytest")
    args = parser.parse_args()
    roux = get_roux(args.roux)
    if args.command == "info":
        return info(roux)
    return verify(roux, args.tests, args.no_tests)


if __name__ == "__main__":
    sys.exit(main())
