import ast
from .. import shape_analysis
from ...passes.ast_utils import is_call, str_to_ast_expr
from .convert_point_wise import PointwiseExprToLoop, Scalarize

class ReductionAndPWExprToLoop(PointwiseExprToLoop):
    def get_reduce_op(self, call_node: ast.Call):
        func = ast.unparse(call_node.func)
        table = {
            'sum': 'sum',
            'min': 'min',
            'max': 'max',
            'np.sum': 'sum',
            'np.min': 'min',
            'np.max': 'max',
            'torch.sum': 'sum',
            'torch.min': 'min',
            'torch.max': 'max',
        }
        return table[func]
    
    def gen_initialization(self, reduce_op, var):
        value = None
        if reduce_op == 'sum':
            value = ast.Constant(0)
        elif reduce_op == 'max':
            value = str_to_ast_expr("float('-inf')")
        elif reduce_op == 'min':
            value = str_to_ast_expr("float('inf')")
        else:
            raise NotImplementedError
        
        return ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=value,
            lineno=None
        )
    
    def rewrite_reduction_assign(self, reduce_op, var, orig_value):
        value = None
        if reduce_op == 'sum':
            value=ast.BinOp(
                op=ast.Add(),
                left=ast.Name(id=var, ctx=ast.Load()),
                right=orig_value.args[0]
            )
        elif reduce_op == 'max' or reduce_op == 'min':
            value=ast.Call(
                func=ast.Name(id=reduce_op, ctx=ast.Load()),
                args=[
                    ast.Name(id=var, ctx=ast.Load()),
                    orig_value.args[0]
                ],
                keywords=[]
            )
        else:
            raise NotImplementedError
        
        return ast.Assign(
            targets=[ast.Name(id=var, ctx=ast.Store())],
            value=value,
            lineno=None
        )
    
    def is_reduction_call(self, node):
        return is_call(node, [
            "np.sum", "np.min", "np.max",
            "torch.sum", "torch.min", "torch.max",
            "sum", "min", "max"
            ]
        )

    def gen_loop(self, node: ast.Assign, low: int|str, up: int|str):
        loop = super().gen_loop(node, low, up)
        # A convenient attribute for APPy
        loop._simd_okay = True
        if self.is_reduction_call(node.value):
            reduce_op = self.get_reduce_op(node.value)
            if not isinstance(node.targets[0], ast.Name):
                raise RuntimeError(f"Only 1D array reduction is supported, but got target: {node.targets[0]}")
            var = node.targets[0].id
            init_stmt = self.gen_initialization(reduce_op, var)
            loop.body = [self.rewrite_reduction_assign(reduce_op, var, node.value)]
            # A convenient attribute for APPy
            loop._reduction = (reduce_op, var)
            return init_stmt, loop
        else:
            return loop
    
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
