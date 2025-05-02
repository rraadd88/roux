from testbook import testbook

@testbook("examples/roux_global_imports.ipynb", execute=True)
def test_roux_global_imports(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_lib_df.ipynb', execute=True)
def test_roux_lib_df(tb):
    assert "78.666667" in tb.cell_output_text('check_na'), tb.cell_output_text('check_na')    
    assert tb.cell_output_text('check_nunique').startswith("species    3\n"), tb.cell_output_text('check_nunique') 
    return

@testbook("examples/roux_lib_dfs.ipynb", execute=True)
def test_roux_lib_dfs(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_lib_df_apply.ipynb", execute=True)
def test_roux_lib_df_apply(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_lib_io.ipynb', execute=True)
def test_roux_lib_io(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_lib_set.ipynb", execute=True)
def test_roux_lib_set(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_lib_str.ipynb', execute=True)
def test_roux_lib_str(tb):
    assert tb.cell_output_text('encoded_long')=='eNqVj00KwjAQRq8Ssqli8QCCK6_gTiSk7WcJNkmZSbRF9OwmjYtuhSwm7_HNz0u2fjCuwyQPQnYUe2E6WYuMWdtxQOalWpnYMMLK_ECxcxY6tvl782TjoDmhV2biI06bElIlVIszQQcLFzaEGwiuxbFKZbXdip0YyVhNs_KkLILm9ExuJ62Z0A1WvtOY-5NVj6CSDawIPYHZeLeM7cnHcYlwS4BT6Y4cemgyuikX_rPU5bwP4HCV7y_fP20r', 'possible change in the funtion.'
    assert tb.cell_output_text('encoded_short')=='e11fafe6bf21d3db843f8a0e4cea21bc600832b3ed738d2b09ee644ce8008e44', 'possible change in the funtion.'
    return 

@testbook('examples/roux_lib_sys.ipynb', execute=True)
def test_roux_lib_sys(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_stat_classify.ipynb", execute=True)
def test_roux_stat_classify(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_stat_cluster.ipynb', execute=True)
def test_roux_stat_cluster(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_stat_corr.ipynb", execute=True)
def test_roux_stat_corr(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_stat_sets.ipynb", execute=True)
def test_roux_stat_sets(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_viz_annot.ipynb', execute=True)
def test_roux_viz_annot(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_viz_ax.ipynb", execute=True)
def test_roux_viz_ax(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_viz_dist.ipynb", execute=True)
def test_roux_viz_dist(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_viz_figure.ipynb", execute=True)
def test_roux_viz_figure(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_viz_io.ipynb', execute=True)
def test_roux_viz_io(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_viz_line.ipynb", execute=True)
def test_roux_viz_line(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_viz_scatter.ipynb", execute=True)
def test_roux_viz_scatter(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook("examples/roux_viz_sets.ipynb", execute=True)
def test_roux_viz_sets(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_viz_theme.ipynb', execute=True)
def test_roux_viz_theme(tb):
    pass # execute only because tests are present in the notebook itself
    return

@testbook('examples/roux_workflow_io.ipynb', execute=True)
def test_roux_workflow_io(tb):
    assert tb.cell_output_text('read_configs')=='value interpolated in config1 = value from metaconfig = value1', tb.cell_output_text('read_configs')
    assert tb.cell_output_text('read_configs_with_inputs')=='value interpolated in config1 = value from metaconfig = modified', tb.cell_output_text('read_configs_with_inputs')

# @testbook("examples/roux_workflow_task.ipynb", execute=True)
# def test_roux_workflow_task(tb):
#   """TODOs: set kernel for testing."""
#     pass # execute only because tests are present in the notebook itself
#     return