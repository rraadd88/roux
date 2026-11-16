"""Checks for the parsing of the notebook parameters (`roux.workflow.pms`)."""

import ast

import pytest

from roux.workflow.pms import extract_pms

PARAMETERS = """## parameters
quote = "it's" # with a comment
backticks = 'a `b` c'
newline = "line1\\nline2"
backslash = 'C:\\\\dir'
both = "say \\"hi\\" it's"
number = 1.5
items = ['x', "y'z"]
none = None
"""


@pytest.mark.parametrize("lines", [PARAMETERS, PARAMETERS.split("\n")])
def test_extract_pms_str_is_valid_code(lines):
    params = extract_pms(lines, fmt="dict")
    assert params["quote"] == "it's"
    assert params["both"] == 'say "hi" it\'s'

    params_args = extract_pms(lines, fmt="str")
    assert len(params_args) == len(params)
    for s in params_args:
        ## before: a SyntaxError for the str values with quotes or newlines
        tree = ast.parse(s)
        key = tree.body[0].targets[0].id
        assert ast.literal_eval(tree.body[0].value) == params[key], s
