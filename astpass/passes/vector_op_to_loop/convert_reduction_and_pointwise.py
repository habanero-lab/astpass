import ast
from .. import shape_analysis
from ...passes.ast_utils import is_call
from .convert_point_wise import PointwiseExprToLoop, Scalarize

class ReductionAndPWExprToLoop(PointwiseExprToLoop):
    def gen_loop(self, node: ast.Assign, low, up):
        if is_call(node.value):
            print(ast.unparse(node))
        return super().gen_loop(node, low, up)
    
def transform(tree, runtime_vals, loop_index_prefix=None):
    """
    Detect and rewrite tensor expressions into explicit loops.

    This pass analyzes pointwise and reduction tensor expressions and 
    rewrites them into explicit loop-based code, assuming all required 
    arrays have already been allocated.

    Parameters
    ----------
    tree : ast.AST
        The input Python AST containing tensor expressions.
    runtime_vals : dict
        A mapping from variable names to runtime values, used for shape
        analysis.
    loop_index_prefix : str, optional
        Prefix to use for generated loop indices.

    Examples
    --------
    ::

        import ast
        import numpy as np
        from astpass.passes import array_expr_to_loop

        tree = ast.parse("c = a + b")
        rt_vals = {
            'a': np.random.randn(10),
            'b': 8,
            'c': np.empty(10)
        }

        tree = array_expr_to_loop.transform(tree, rt_vals)
        print(ast.unparse(tree))

    Notes
    -----
    This pass assumes that all arrays have been allocated beforehand.
    It only generates loop constructs and does not perform any memory
    allocation. All variables appearing in the input code are assumed
    to be already defined. If the input code inherently requires memory
    allocation for intermediate results, an exception will be raised.
    """
    shape_info = shape_analysis.analyze(tree, runtime_vals)
    return ReductionAndPWExprToLoop(shape_info, loop_index_prefix).visit(tree)
