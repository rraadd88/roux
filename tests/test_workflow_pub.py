"""`to_pub` (roux/workflow/pub.py): the figures, the analyses and the plan (dry run) on a toy repository."""

import subprocess

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


def test_read_figs(tmp_path):
    from roux.workflow.pub import read_figs

    _touch_plots(tmp_path)
    df = read_figs(
        FIGS,
        tag="v2",
        tag_pattern=r"(?<=outputs/)v\d+(?=/)",
        input_path=str(tmp_path),
    )
    assert df["fig"].tolist() == ["1C1", "1D1", "1D2", "2A1", "S211", "S221"]
    assert df["script label"].tolist() == ["a", "b_1", "b_2", "q", "s_1", "s_2"]
    assert df["plot label"].tolist() == [
        "FIG1C",
        "FIG1D plot1",
        "FIG1D plot2",
        "FIG2",
        "FIGS2 plot1",
        "FIGS2 plot2",
    ]
    ## the version: `tag`
    assert df["input path"].tolist()[0] == f"{tmp_path}/outputs/v2/c"
    assert df["exists"].all()


def test_read_figs_checks(tmp_path):
    from roux.workflow.pub import read_figs

    ## missing plots
    with pytest.raises(ValueError, match="not found: 1C"):
        read_figs(FIGS, input_path=str(tmp_path))
    assert not read_figs(FIGS, input_path=str(tmp_path), check=False)["exists"].any()
    ## the script names are the file names of the paper repository: unique
    with pytest.raises(ValueError, match=r"not unique.*\['a'\]"):
        read_figs(
            {
                1: {
                    "A": {"script": "a", "plots": ["x"]},
                    "B": {"script": "a", "plots": ["y"]},
                }
            },
            check=False,
        )
    ## the short forms (a path or a list instead of {script, plots})
    with pytest.raises(AssertionError, match="expected"):
        read_figs({1: {"A": ["x"]}}, check=False)
    with pytest.raises(AssertionError, match="2 scripts for 1 plots"):
        read_figs({1: {"A": {"script": ["a", "b"], "plots": ["x"]}}}, check=False)


def test_get_analyses():
    from roux.workflow.pub import _get_analyses

    assert _get_analyses({"A": [1], "B": [], "C": [2, 3]}) == {
        "1": "A",
        "S1": "A",
        "2": "B",
        "3": "C",
        "S2": "C",
        "S3": "C",
    }
    with pytest.raises(AssertionError, match="S1 in two analyses"):
        _get_analyses({"A": [1], "B": [1]})


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


def test_to_pub_figure_without_analysis(analysis_repo, tmp_path):
    import yaml

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
    cfg = yaml.safe_load((analysis_repo / "to_pub.cfg").read_text())
    cfg["ana_to_sis"] = {"Analysis A": []}
    (analysis_repo / "to_pub.cfg").write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match=r"without an analysis.*\['2', 'S2'\]"):
        to_pub(
            input_path=str(analysis_repo / "to_pub.cfg"),
            dry_run="True",
            allow_dirty="True",
        )


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


def test_clean_plot_script():
    from roux.workflow.pub import _clean_plot_script

    text = _clean_plot_script(PLOT_PY)
    assert "# a comment" not in text and "#noqa" not in text
    assert 'ax.set_title("#1")' in text
    assert "plotp=__file__," in text and "/an/old/path" not in text
    assert "\n\n\n" not in text
    assert not any(s != s.rstrip() for s in text.split("\n"))
    compile(text, "plot.py", "exec")


def test_to_panel(tmp_path):
    from roux.lib.io import read_dict, read_table
    from roux.workflow.pub import _to_panel

    _save_plot(tmp_path / "outputs/v1/c.pdf")
    review = tmp_path / "review/c"
    assert _to_panel(str(tmp_path / "outputs/v1/c"), str(review)) == {"n": 3}
    assert sorted(p.name for p in review.iterdir()) == [
        "config.yaml",
        "data.pqt",
        "plot.py",
        "stats.yaml",
    ]
    assert read_table(str(review / "data.pqt")).shape == (3, 2)
    assert read_dict(str(review / "config.yaml")) == {"color": "k"}
    assert "plotp=__file__," in (review / "plot.py").read_text()
    ## not remade: the saved stats
    mtime = (review / "data.pqt").stat().st_mtime_ns
    assert _to_panel(str(tmp_path / "outputs/v1/c"), str(review)) == {"n": 3}
    assert (review / "data.pqt").stat().st_mtime_ns == mtime


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
    text = open(outp).read()
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


def test_to_pub_review_files(tmp_path, caplog):
    import yaml

    from roux.workflow.pub import to_pub

    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / ".gitignore").write_text("outputs/\nresults/\n")
    figs = {
        1: {"A": {"script": "a", "plots": ["outputs/v1/a1.pdf", "outputs/v1/a2.pdf"]}}
    }
    cfg = dict(
        input_path=".",
        output_path="results/latest.yaml",
        pub_path="../proj-pub",
        figs=figs,
        ana_to_sis={"Analysis A": []},
    )
    (repo / "to_pub.cfg").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(
        repo,
        "worktree",
        "add",
        "-q",
        "--orphan",
        "-b",
        "pub",
        str(tmp_path / "proj-pub"),
    )
    for name in ["a1", "a2"]:
        _save_plot(repo / f"outputs/v1/{name}.pdf")

    to_pub(input_path=str(repo / "to_pub.cfg"))
    review = repo / "results/latest"
    assert sorted(
        str(p.relative_to(review)) for p in review.rglob("*") if p.is_file()
    ) == [
        "figs.tsv",
        "figs/fig1A1.pdf",
        "figs/fig1A2.pdf",
        *[
            f"repo/figs/fig1A{i}/{f}"
            for i in [1, 2]
            for f in ["config.yaml", "data.pqt", "plot.py", "stats.yaml"]
        ],
    ]
    assert "not written yet" in caplog.text
