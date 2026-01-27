import ast
import textwrap
from astpass.passes import normalize_ranges

def test1():
    code = """
    range(10)
    """
    tree = ast.parse(textwrap.dedent(code))
    new_tree = normalize_ranges.transform(tree)
    new_code = ast.unparse(new_tree)

    expected = """
    range(0, 10, 1)
    """
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test2():
    code = """
    range(1, 10)
    """
    tree = ast.parse(textwrap.dedent(code))
    new_tree = normalize_ranges.transform(tree)
    new_code = ast.unparse(new_tree)

    expected = """
    range(1, 10, 1)
    """
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test3():
    code = """
    range(1, 10, 2)
    """
    tree = ast.parse(textwrap.dedent(code))
    new_tree = normalize_ranges.transform(tree)
    new_code = ast.unparse(new_tree)

    expected = """
    range(1, 10, 2)
    """
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))