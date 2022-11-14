from testbook import testbook

@testbook('examples/roux_lib_str.ipynb', execute=True)
def test_stdout(tb):
    assert tb.cell_output_text('encoded_long')=='eNqVj00KwjAQRq8Ssqli8QCCK6_gTiSk7WcJNkmZSbRF9OwmjYtuhSwm7_HNz0u2fjCuwyQPQnYUe2E6WYuMWdtxQOalWpnYMMLK_ECxcxY6tvl782TjoDmhV2biI06bElIlVIszQQcLFzaEGwiuxbFKZbXdip0YyVhNs_KkLILm9ExuJ62Z0A1WvtOY-5NVj6CSDawIPYHZeLeM7cnHcYlwS4BT6Y4cemgyuikX_rPU5bwP4HCV7y_fP20r', 'possible change in the funtion.'
    assert tb.cell_output_text('encoded_short')=='e11fafe6bf21d3db843f8a0e4cea21bc600832b3ed738d2b09ee644ce8008e44', 'possible change in the funtion.'
    
@testbook('examples/roux_lib_sys.ipynb', execute=True)
def test_stdout(tb):
    assert tb.cell_output_text('to_output_paths')=="7", "not all inputs and paths processed"

@testbook('examples/roux_lib_df.ipynb', execute=True)
def test_stdout(tb):
    # assert tb.cell_output_text(1) == 'hello world!'
    assert 'False' in tb.cell_output_text('validate_no_dups') # 0-based
    assert 'sepal_length_x' in tb.cell_output_text('merge') # 0-based
    
@testbook('examples/roux_lib_io.ipynb', execute=True)
def test_stdout(tb):
    assert "data/table.tsv" in tb.cell_output_text('to_table')

@testbook('examples/roux_query.ipynb', execute=True)
def test_stdout(tb):
    assert "data/biomart/00_raw.tsv" in tb.cell_output_text('to_table')    

@testbook('examples/roux_stat_cluster.ipynb', execute=True)
def test_stdout(tb):
    assert 'data/biomart/01_dedup.tsv' in tb.cell_output_text('to_table')    

@testbook('examples/roux_viz_annot.ipynb', execute=True)
def test_stdout(tb):
    assert 'data/biomart/01_dedup.tsv' in tb.cell_output_text('to_table')    

@testbook('examples/roux_viz_io.ipynb', execute=True)
def test_stdout(tb):
    assert "title={'center':'modified'}" in tb.cell_output_text('read_plot_modified')
    
@testbook('examples/roux_global_imports.ipynb', execute=True)
def test_stdout(tb):
    assert int(tb.cell_output_text('functions_from_roux').split('=')[1].replace('.','')) < 250