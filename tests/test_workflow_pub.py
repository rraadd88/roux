"""`to_pub` (roux/workflow/pub.py): the figures, the analyses and the plan (dry run) on a toy repository."""

import subprocess
from pathlib import Path

import pytest

FIGS = {
    1: {
        "C": {"script": "a", "plots": ["outputs/v1/c.pdf"]},
        "D": {
            "script": "b",
            "plots": ["outputs/v1/d1.pdf", "outputs/v1/d2.pdf"],
        },
    },
    2: {"A": {"plots": ["outputs/v1/q.pdf"]}},
    ## a figure without panels
    "S2": {
        "script": "s",
        "plots": ["outputs/v1/s1.pdf", "outputs/v1/s2.pdf"],
    },
}


def _touch_plots(root, tag="v2"):
    for name in ["c", "d1", "d2", "q", "s1", "s2"]:
        d = root / "outputs" / tag / name
        d.mkdir(parents=True)
        d.with_suffix(".pdf").write_text("")


def _git(repo, *args):
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def analysis_repo(tmp_path):
    import yaml

    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / ".gitignore").write_text("outputs/\n")
    (repo / "README.md").write_text("# proj\n")
    cfg = dict(
        input_path=".",
        output_path="results/latest.yaml",
        pub_path="../proj-pub",
        tag="v2",
        tag_pattern=r"(?<=outputs/)v\d+(?=/)",
        figs=FIGS,
        ana_to_sis={"Analysis A": [2], "Analysis B": []},
        copy={"README.md": "README.md"},
    )
    (repo / "to_pub.cfg").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _touch_plots(repo)
    return repo


def test_to_pub_needs_the_worktree(analysis_repo, capsys):
    from roux.workflow.pub import to_pub

    with pytest.raises(ValueError, match="to be made by the user"):
        to_pub(input_path=str(analysis_repo / "to_pub.cfg"), dry_run="True")
    assert "worktree add --orphan -b pub" in capsys.readouterr().out


def test_to_pub_dry_run(analysis_repo, tmp_path, capsys):
    from roux.workflow.pub import to_pub

    _git(
        analysis_repo,
        "worktree",
        "add",
        "-q",
        "--orphan",
        "-b",
        "pub",
        str(tmp_path / "proj-pub"),
    )
    ## the paths of the params file are relative to its directory, not to the current one
    assert to_pub(input_path=str(analysis_repo / "to_pub.cfg"), dry_run="True") is None
    lines = capsys.readouterr().out.splitlines()
    root = analysis_repo
    assert lines == [
        f"panel FIG1C {root}/outputs/v2/c -> scripts/analysis_a/a.py configs/analyses/analysis_a/a.yaml data/analyses/analysis_a/a.pqt",
        f"panel FIG1D plot1 {root}/outputs/v2/d1 -> scripts/analysis_a/b_1.py configs/analyses/analysis_a/b_1.yaml data/analyses/analysis_a/b_1.pqt",
        f"panel FIG1D plot2 {root}/outputs/v2/d2 -> scripts/analysis_a/b_2.py configs/analyses/analysis_a/b_2.yaml data/analyses/analysis_a/b_2.pqt",
        f"panel FIG2 {root}/outputs/v2/q -> scripts/analysis_b/q.py configs/analyses/analysis_b/q.yaml data/analyses/analysis_b/q.pqt",
        f"panel FIGS2 plot1 {root}/outputs/v2/s1 -> scripts/analysis_a/s_1.py configs/analyses/analysis_a/s_1.yaml data/analyses/analysis_a/s_1.pqt",
        f"panel FIGS2 plot2 {root}/outputs/v2/s2 -> scripts/analysis_a/s_2.py configs/analyses/analysis_a/s_2.yaml data/analyses/analysis_a/s_2.pqt",
        f"copy {root}/README.md -> README.md",
        "write configs/analyses.yaml",
        "write pyproject.toml",
        "write .gitignore",
    ]
    ## nothing written
    assert not (analysis_repo / "results").exists()
    assert sorted(p.name for p in (tmp_path / "proj-pub").iterdir()) == [".git"]


## S3: the panels (review files) and the scripts of the paper repository
PLOT_PY = """from pathlib import Path
import matplotlib.pyplot as plt
def plot_(
    plotp="/an/old/path",
    data=None,
    df1=None,
    kws_plot=None,
    ax=None,
    fig=None,
    outd=None,
    fun_data=None,
    **kws_set,
    ):

    ## get the inputs
    from roux.viz.io import get_plot_inputs
    plotp,data,kws_plot=get_plot_inputs(plotp=plotp,df1=data,kws_plot=kws_plot,outd=f"{Path(__file__).parent}");
    data=fun_data(data) if not fun_data is None else data;


    ## plotting
    if ax is None:fig,ax=plt.subplots() #noqa
    ax.plot(data["x"], data["y"], **kws_plot)  # a comment
    ax.set_title("#1")
    ax.stats = {"n": len(data)}
    ax.set(**kws_set)
    return ax
"""


def _save_plot(plot_path):
    """A plot as saved by `roux.viz.io.to_plot`: `<plot>.pdf` and `<plot>/{data.pqt,config.yaml,plot.py}`."""
    import pandas as pd
    import yaml

    d = plot_path.with_suffix("")
    d.mkdir(parents=True)
    pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]}).to_parquet(d / "data.pqt")
    (d / "config.yaml").write_text(yaml.safe_dump({"color": "k"}))
    (d / "plot.py").write_text(PLOT_PY)
    plot_path.write_text("")


def test_to_pub_script(tmp_path):
    from roux.workflow.pub import _to_pub_script

    (tmp_path / "helpers").mkdir()
    (tmp_path / "helpers/__init__.py").write_text("")
    (tmp_path / "helpers/mod.py").write_text(
        "import math\n\n\ndef f(x):\n    return math.sqrt(x)\n\n\ndef g():\n    return f(1)\n"
    )
    script = PLOT_PY.replace(
        "    ## plotting\n",
        "    ## plotting\n    try:\n        import sys\n        sys.path.append('../')\n"
        "        from helpers.mod import f\n    except ImportError:\n        pass\n",
    )
    (tmp_path / "plot.py").write_text(script)
    outp = _to_pub_script(
        str(tmp_path / "plot.py"),
        str(tmp_path / "out/a.py"),
        src_dir_path=str(tmp_path),
        inline=["helpers"],
    )
    text = Path(outp).read_text()
    compile(text, outp, "exec")
    assert "from helpers" not in text and "sys.path.append" not in text
    assert "    import math\n    ## from the source: helpers.mod\n    def f(x):" in text
    assert "def g" not in text
    assert "def run(" in text and "kws_plot" not in text and "config" in text
    assert "get_plot_inputs" not in text
    ## without `inline`: the imports are kept
    _to_pub_script(
        str(tmp_path / "plot.py"),
        str(tmp_path / "out/b.py"),
        src_dir_path=str(tmp_path),
    )
    assert "from helpers.mod import f" in (tmp_path / "out/b.py").read_text()
    ## a function that needs another function of its module (not copied)
    (tmp_path / "plot.py").write_text(script.replace("import f\n", "import g\n"))
    with pytest.raises(AssertionError):
        _to_pub_script(
            str(tmp_path / "plot.py"),
            str(tmp_path / "out/c.py"),
            src_dir_path=str(tmp_path),
            inline="helpers",
        )


def test_to_pub(tmp_path, caplog, capsys):
    """A full run: the review files, then the paper repository; a rerun lists the stale files."""
    import json

    import yaml

    from roux.lib.io import read_dict
    from roux.workflow.pub import to_pub

    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / ".gitignore").write_text("outputs/\nresults/\nscripts\n")
    (repo / "README.md").write_text("# proj\n")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "proj"\n\n[tool.uv.sources]\nlib = { path = "../lib" }\n\n[tool.ruff]\nline-length = 88\n'
    )
    figs = {
        1: {"A": {"script": "a", "plots": ["outputs/v1/a1.pdf", "outputs/v1/a2.pdf"]}}
    }
    cfg = dict(
        input_path=".",
        output_path="results/latest.yaml",
        pub_path="../proj-pub",
        figs=figs,
        ana_to_sis={"Analysis A": []},
        copy={"README.md": "README.md"},
        gitignore=["!scripts/"],
        cols_desc={"x": "the x values", "z": "not a column"},
    )
    (repo / "to_pub.cfg").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    out = tmp_path / "proj-pub"
    _git(repo, "worktree", "add", "-q", "--orphan", "-b", "pub", str(out))
    for name in ["a1", "a2"]:
        _save_plot(repo / f"outputs/v1/{name}.pdf")

    assert to_pub(input_path=str(repo / "to_pub.cfg")).endswith("results/latest.yaml")

    ## review files
    review = repo / "results/latest"
    files = {str(p.relative_to(review)) for p in review.rglob("*") if p.is_file()}
    assert {
        "figs.tsv",
        "figs/fig1A1.pdf",
        "figs/fig1A2.pdf",
        "stats.yaml",
        "stats_fmt.json",
        "tables/source_data.xlsx",
        *[
            f"repo/figs/fig1A{i}/{f}"
            for i in [1, 2]
            for f in ["config.yaml", "data.pqt", "plot.py", "stats.yaml"]
        ],
    } <= files
    assert read_dict(str(review / "stats.yaml")) == {"1A1:n": 3, "1A2:n": 3}
    assert json.loads((review / "stats_fmt.json").read_text()) == {
        "{{1A1:n}}": "3",
        "{{1A2:n}}": "3",
    }

    ## the paper repository
    pub_files = [
        "README.md",
        "configs/analyses.yaml",
        "configs/analyses/analysis_a/a_1.yaml",
        "configs/analyses/analysis_a/a_2.yaml",
        "data/analyses/analysis_a/a_1.pqt",
        "data/analyses/analysis_a/a_2.pqt",
        "pyproject.toml",
        "scripts/analysis_a/a_1.py",
        "scripts/analysis_a/a_2.py",
        ".gitignore",
    ]
    assert sorted(
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and ".git" != p.name
    ) == sorted(pub_files)
    assert read_dict(str(out / "configs/analyses.yaml")) == {
        "Analysis A": ["a_1", "a_2"]
    }
    script = (out / "scripts/analysis_a/a_1.py").read_text()
    assert "def run(" in script and "get_plot_inputs" not in script
    pyproject = (out / "pyproject.toml").read_text()
    assert "[tool.uv.sources]" not in pyproject and "../lib" not in pyproject
    assert "[project]" in pyproject and "[tool.ruff]" in pyproject
    assert (out / ".gitignore").read_text().startswith("outputs/\nresults/\nscripts\n")
    assert (out / ".gitignore").read_text().endswith("\n!scripts/\n")
    ## the commit: printed, not run
    printed = capsys.readouterr().out
    assert "add -A" in printed and "commit -m" in printed
    assert _git_out(out, "status", "--porcelain") != ""
    assert (
        subprocess.run(
            ["git", "-C", str(out), "rev-parse", "--verify", "-q", "HEAD"]
        ).returncode
        == 1
    )
    ## no temporary directory left
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    ## the marker
    marker = read_dict(str(repo / "results/latest.yaml"))
    assert sorted(marker["files"]) == sorted(pub_files)
    assert marker["pub"]["branch"] == "pub"

    ## rerun: the panels are not remade, the hand-made and stale files are kept (listed)
    (out / "NOTES.md").write_text("by hand\n")
    (out / "scripts/analysis_a/old.py").write_text("")
    to_pub(input_path=str(repo / "to_pub.cfg"))
    assert "stale?): ['scripts/analysis_a/old.py']" in caplog.text
    assert (out / "NOTES.md").exists() and (out / "scripts/analysis_a/old.py").exists()


def test_to_review(tmp_path, monkeypatch):
    """The review files only: no git repository, no paper repository."""
    import yaml

    from roux.lib.io import read_dict
    from roux.workflow.pub import to_review

    repo = tmp_path / "proj"
    repo.mkdir()
    cfg = dict(
        input_path=".",
        output_path="results/latest.yaml",
        pub_path="../proj-pub",
        figs={1: {"A": {"script": "a", "plots": ["outputs/v1/a1.pdf"]}}},
        ana_to_sis={"Analysis A": []},
        copy={"README.md": "README.md"},
    )
    (repo / "to_pub.cfg").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _save_plot(repo / "outputs/v1/a1.pdf")
    (repo / "outputs/v1/a1.png").write_text("")

    monkeypatch.chdir(repo)
    df = to_review("to_pub.cfg")
    assert df["plot label"].tolist() == ["FIG1"]
    review = repo / "results/latest"
    assert {
        "figs.tsv",
        "figs/fig1.pdf",
        "figs/fig1.png",
        "stats.yaml",
        "tables/source_data.xlsx",
        "repo/figs/fig1/data.pqt",
    } <= {str(p.relative_to(review)) for p in review.rglob("*") if p.is_file()}
    assert read_dict(str(review / "stats.yaml")) == {"1:n": 3}
    assert (
        not (tmp_path / "proj-pub").exists()
        and not (repo / "results/latest.yaml").exists()
    )


def _git_out(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def test_format_number():
    from roux.lib.str import format_number

    _vals = [
        0.00000002,
        0.002367729,
        -0.23677299175778388,
        562.0,
        23677299.0,
        3.9220394750515513e-219,
        1.352054248160119e-07,
    ]
    assert [format_number(x) for x in _vals] == [
        "2e-8",
        "0.002",
        "-0.24",
        "562",
        "23.7e6",
        "3e-219",
        "1e-7",
    ]
    assert [format_number(x) for x in [7, -3, 2000000, 0.5]] == [
        "7",
        "-3",
        "2e6",
        "0.5",
    ]


def test_read_cfg_interpolation(tmp_path):
    """TODO 51: OmegaConf interpolations resolved, with the given inputs; a missing value raises."""
    from roux.workflow.pub import _read_cfg

    p = tmp_path / "to_pub.cfg"
    p.write_text("tag: ???\nreview_path: processed/${tag}/review\n")
    assert _read_cfg(str(p), inputs=dict(tag="v2"))["review_path"].endswith(
        "processed/v2/review"
    )
    with pytest.raises(Exception, match="tag"):
        _read_cfg(str(p))
