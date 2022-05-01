from testbook import testbook

@testbook('examples/lib_roux_data.ipynb', execute=True)
def test_stdout(tb):
    # assert tb.cell_output_text(1) == 'hello world!'
    assert 'False' in tb.cell_output_text(2)