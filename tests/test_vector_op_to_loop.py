import ast
import textwrap
import numpy as np

from astpass.passes import vector_op_to_loop

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
    tree = vector_op_to_loop.transform(tree, rt_vals)

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
    tree = vector_op_to_loop.transform(tree, rt_vals)

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
    tree = vector_op_to_loop.transform(tree, rt_vals)

    expected = """
    for __i0 in range(0, 10):
        c[__i0] = a[__i0] + b
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_add4():
    code = """
    c[:] = a[:] * 2 + b
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'b': 1.0,
        'c': np.empty(10)
    }
    tree = vector_op_to_loop.transform(tree, rt_vals)

    expected = """
    for __i0 in range(0, 10):
        c[__i0] = a[__i0] * 2 + b
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_np_sum1():
    code = """
    c = np.sum(a)
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'c': 0.0,
        'np': np
    }
    tree = vector_op_to_loop.transform(tree, rt_vals)

    expected = """
    c = 0
    for __i0 in range(0, 10):
        c = c + a[__i0]
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_np_sum2():
    code = """
    c = np.sum(a * b)
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'b': np.random.randn(10),
        'c': 0.0,
        'np': np
    }
    tree = vector_op_to_loop.transform(tree, rt_vals)

    expected = """
    c = 0
    for __i0 in range(0, 10):
        c = c + a[__i0] * b[__i0]
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_np_max1():
    code = """
    c = np.max(a)
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'c': 0.0,
        'np': np
    }
    tree = vector_op_to_loop.transform(tree, rt_vals)

    expected = """
    c = float('-inf')
    for __i0 in range(0, 10):
        c = max(c, a[__i0])
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))

def test_np_min1():
    code = """
    c = np.min(a)
    """
    tree = ast.parse(textwrap.dedent(code))
    rt_vals = {
        'a': np.random.randn(10),
        'c': 0.0,
        'np': np
    }
    tree = vector_op_to_loop.transform(tree, rt_vals)

    expected = """
    c = float('inf')
    for __i0 in range(0, 10):
        c = min(c, a[__i0])
    """
    new_code = ast.unparse(tree)
    assert new_code == ast.unparse(ast.parse(textwrap.dedent(expected)))