import ast
import textwrap
import numpy as np

from astpass.passes import array_expr_to_loop

def test_add1():
    code = """
    c = a + b
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'b': 1.0,
        'c': np.empty(10)
    }
    tree = array_expr_to_loop.transform(tree, rt_vals)

    expected = """
    for __i0 in range(0, 10):
        c[__i0] = a[__i0] + b
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_add2():
    code = """
    c = a[:] + b
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'b': 1.0,
        'c': np.empty(10)
    }
    tree = array_expr_to_loop.transform(tree, rt_vals)

    expected = """
    for __i0 in range(0, 10):
        c[__i0] = a[__i0] + b
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_add3():
    code = """
    c[:] = a[:] + b
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'b': 1.0,
        'c': np.empty(10)
    }
    tree = array_expr_to_loop.transform(tree, rt_vals)

    expected = """
    for __i0 in range(0, 10):
        c[__i0] = a[__i0] + b
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_add4():
    code = """
    c = a + b
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(3, 4),
        'b': 1.0,
        'c': np.empty((3, 4))
    }
    tree = array_expr_to_loop.transform(tree, rt_vals)

    expected = """
    for __i0 in range(0, 3):
        for __i1 in range(0, 4):
            c[__i0, __i1] = a[__i0, __i1] + b
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))