import ast
import textwrap
import astpass as at

def test_get_used_names():
    code = """
    for i in range(N):
        c[i] = a[i] + b[i]
    """
    tree = ast.parse(textwrap.dedent(code))
    names = at.get_used_names(tree, no_funcname=False)
    assert names == ['i', 'range', 'N', 'c', 'a', 'b']

    names = at.get_used_names(tree, no_funcname=True)
    assert names == ['i', 'N', 'c', 'a', 'b']
