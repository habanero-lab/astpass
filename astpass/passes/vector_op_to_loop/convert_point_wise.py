import ast
from ...passes import shape_analysis

def str_to_ast_expr(expr_str):
    return ast.parse(expr_str).body[0].value

class CollectNonzeroShapes(ast.NodeVisitor):
    def __init__(self, shape_info):
        self.shape_info = shape_info
        self.nonzero_shapes = []

    def get_node_shape(self, node):
        if node not in self.shape_info:
            raise KeyError(f"Shape info not found for node {type(node)}: {ast.unparse(node)}")
        return self.shape_info[node]

    def visit_Subscript(self, node):
        shape = self.get_node_shape(node)
        assert shape is not None
        if len(shape) > 0:
            self.nonzero_shapes.append(shape)

    def visit_Call(self, node):
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node):
        shape = self.get_node_shape(node)
        assert shape is not None
        if len(shape) > 0:
            self.nonzero_shapes.append(shape)

class Scalarize(ast.NodeTransformer):
    def __init__(self, shape_info, idx):
        self.shape_info = shape_info
        self.idx = idx

    def get_node_shape(self, node):
        if node not in self.shape_info:
            raise KeyError(f"Shape info not found for node {type(node)}: {ast.unparse(node)}")
        return self.shape_info[node]
    
    def visit_Call(self, node):
        for arg in node.args:
            self.visit(arg)
        return node

    def visit_Name(self, node):
        shape = self.get_node_shape(node)
        if len(shape) > 0:
            assert len(shape) == 1
            return ast.Subscript(
                value=ast.Name(id=node.id, ctx=ast.Load()),
                slice=ast.Name(id=self.idx, ctx=ast.Load()),
                ctx=ast.Load()
            )
        else:
            return node
        
    def visit_Subscript(self, node):
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
        num_slices = sum([isinstance(idx, ast.Slice) for idx in indices])
        assert num_slices in [0, 1], f"A subscript should have 0 or 1 sliced indices, but got {num_slices}"

        new_scalar_index = ast.Name(id=self.idx, ctx=ast.Load())
        if num_slices == 0:
            new_indices = indices + new_scalar_index
        elif num_slices == 1:
            new_indices = [new_scalar_index if isinstance(idx, ast.Slice) else idx for idx in indices]

        return ast.Subscript(
                value=node.value,
                slice=ast.Tuple(elts=new_indices, ctx=ast.Load()) if len(new_indices) > 1 else new_indices[0],
                ctx=ast.Load()
            )


class PointwiseExprToLoop(ast.NodeTransformer):
    def __init__(self, shape_info, loop_index_prefix=None):
        self.shape_info = shape_info
        self.loop_index_prefix = loop_index_prefix if loop_index_prefix is not None else "__i"
        self.loop_index_count = 0

    def get_new_loop_index(self):
        name = f"{self.loop_index_prefix}{self.loop_index_count}"
        self.loop_index_count += 1
        return name

    def get_loop_bounds(self, shapes):
        if not all([s == shapes[0] for s in shapes]):
            raise RuntimeError(f"Shapes are not the same: {shapes}")
        
        if not all([len(s) == 1 for s in shapes]):
            raise RuntimeError(f"Only 1D array expansion is supported, but got shapes: {shapes}")
        
        bound = shapes[0][0]
        assert isinstance(bound, (int, str))
        if isinstance(bound, int):
            low, up = 0, bound
        elif isinstance(bound, str):
            low, up = bound.split(":")
        return low, up

    def visit_Assign(self, node):        
        shape_visitor = CollectNonzeroShapes(self.shape_info)
        shape_visitor.visit(node)
        nonzero_shapes = shape_visitor.nonzero_shapes
        if nonzero_shapes:
            low, up = self.get_loop_bounds(nonzero_shapes)
            return self.gen_loop(node, low, up)
        else:
            return node

    def gen_loop(self, node, low, up):
        index = self.get_new_loop_index()
        loop = ast.For(
            target=ast.Name(id=index, ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id='range', ctx=ast.Load()),
                args=[
                    str_to_ast_expr(low) if isinstance(low, str) else ast.Constant(low),
                    str_to_ast_expr(up) if isinstance(up, str) else ast.Constant(up)
                ],
                keywords=[]
            ),
            body=[Scalarize(self.shape_info, index).visit(node)],
            orelse=[],
            lineno=node.lineno
        )
        return loop

def transform(tree, runtime_vals, loop_index_prefix=None):
    '''
    This pass detects and rewrites tensor expressions to explicit loops.

    Note 
    ----
    This pass assumes that all arrays have been allocated, and it only generates
    the loops, no memory allocations will be performed. In other words, all variables
    appeared in the input code should already be defined.
    '''
    shape_info = shape_analysis.analyze(tree, runtime_vals)
    return PointwiseExprToLoop(shape_info, loop_index_prefix).visit(tree)