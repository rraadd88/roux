from testbook import testbook

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
