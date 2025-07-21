import logging
logging.basicConfig(level='INFO',force=True)

from pathlib import Path

## fix ipython
import subprocess

r = subprocess.run(
        "ipython profile create",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
logging.info(r.stdout)
r = subprocess.run(
        "ipython locate",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
logging.info(r.stdout)
ipy_path=Path(r.stdout.split('\n')[0]).absolute().as_posix()
assert Path(ipy_path).exists(), ipy_path

r = subprocess.run(
        f"echo 'c.HistoryManager.enabled = False\nc.HistoryAccessor.enabled = False' >> {ipy_path}/profile_default/ipython_config.py",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
logging.info(r)

output_dir_path='examples/outputs/'
Path(output_dir_path).mkdir(parents=True,exist_ok=True)

def run_test(notebook_path, output_path=None):
    import papermill as pm
    # Execute the notebook and save the output
    pm.execute_notebook(
        notebook_path,
        f"{output_dir_path}/{Path(notebook_path).name}",
        kernel_name="python3",  # or change to your kernel
        start_timeout = 600,
    )

def test_roux_global_imports(
    p="examples/roux_global_imports.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_lib_df(
    p="examples/roux_lib_df.ipynb",
    ):
    return run_test(
        p,
    )

def test_roux_lib_dfs(
    p="examples/roux_lib_dfs.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_lib_df_apply(
    p="examples/roux_lib_df_apply.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_lib_io(
    p="examples/roux_lib_io.ipynb",
    ):
    return run_test(
        p,
    )

def test_roux_lib_set(
    p="examples/roux_lib_set.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_lib_str(
    p="examples/roux_lib_str.ipynb",
    ):
    return run_test(
        p,
    ) 

# def test_roux_lib_sys(
#     p="examples/roux_lib_sys.ipynb",
#     ):
#     return run_test(
#         p,
#     )

# def test_roux_stat_classify(
#     p="examples/roux_stat_classify.ipynb"
#     ):
#     return run_test(
#         p,
#     )

def test_roux_stat_cluster(
    p="examples/roux_stat_cluster.ipynb",
    ):
    return run_test(
        p,
    )

def test_roux_stat_corr(
    p="examples/roux_stat_corr.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_stat_sets(
    p="examples/roux_stat_sets.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_viz_annot(
    p="examples/roux_viz_annot.ipynb",
    ):
    return run_test(
        p,
    )

# def test_roux_viz_ax(
#     p="examples/roux_viz_ax.ipynb"
#     ):
#     return run_test(
#         p,
#     )

def test_roux_viz_dist(
    p="examples/roux_viz_dist.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_viz_figure(
    p="examples/roux_viz_figure.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_viz_io(
    p="examples/roux_viz_io.ipynb",
    ):
    return run_test(
        p,
    )

def test_roux_viz_line(
    p="examples/roux_viz_line.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_viz_scatter(
    p="examples/roux_viz_scatter.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_viz_sets(
    p="examples/roux_viz_sets.ipynb"
    ):
    return run_test(
        p,
    )

def test_roux_viz_theme(
    p="examples/roux_viz_theme.ipynb",
    ):
    return run_test(
        p,
    )

def test_roux_workflow_io(
    p="examples/roux_workflow_io.ipynb",
    ):
    return run_test(
        p,
    )