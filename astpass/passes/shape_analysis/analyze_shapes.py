import ast
import inspect
from . import func_table
from ..ast_utils import is_call

class AnalyzeExprShapes(ast.NodeVisitor):
    def __init__(self, rt_vals):
        self.node_shapes = {}
        self.var_shapes = {}
        self.modules = {}
        self.init_rt_var_shapes(rt_vals)
        self.init_module_names(rt_vals)

    def init_rt_var_shapes(self, rt_vals):
        for var, val in rt_vals.items():
            if isinstance(val, (int, float, bool)):
                self.var_shapes[var] = ()
            elif hasattr(val, 'shape'):
                self.var_shapes[var] = val.shape
            else:
                raise RuntimeError(f"Unsupported type: {type(val)}")

    def init_module_names(self, rt_vals):
        for var, val in rt_vals.items():
            if inspect.ismodule(val):
                self.modules[var] = val

    # Two types of leaf nodes
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, bool)):
            self.node_shapes[node] = ()
        else:
            raise RuntimeError(f"Unsupported constant type: {type(node.value)}")

    def visit_Name(self, node):
        if node.id in self.var_shapes:
            self.node_shapes[node] = self.var_shapes[node.id]
        else:
            raise RuntimeError(f"Name {node.id} not found in runtime values")
    
    # Five ways to combine nodes
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        f = getattr(func_table, 'uop_generic')
        self.node_shapes[node] = f(self.node_shapes[node.operand])
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.BitAnd, ast.BitOr, ast.BitXor)):
            f = getattr(func_table, 'binop_generic')
            self.node_shapes[node] = f(self.node_shapes[node.left], self.node_shapes[node.right])
        elif isinstance(node.op, ast.MatMult):
            f = getattr(func_table, 'matmul_generic')
            self.node_shapes[node] = f(self.node_shapes[node.left], self.node_shapes[node.right])
        else:
            raise NotImplementedError(f"Binary operator {node.op} not implemented")
        
    def visit_Compare(self, node):
        self.generic_visit(node)
        assert len(node.comparators) == 1
        f = getattr(func_table, 'compare_generic')        
        self.node_shapes[node] = f(self.node_shapes[node.left], self.node_shapes[node.comparators[0]])

    def visit_IfExp(self, node: ast.IfExp):
        self.generic_visit(node)
        f = getattr(func_table, 'ifexp_generic')
        self.node_shapes[node] = f(self.node_shapes[node.test], self.node_shapes[node.body], self.node_shapes[node.orelse])

    def dispatch_call(self, f_name, args):
        if f_name in ['numpy_sum', 'numpy_min', 'numpy_max', 'numpy_argmin', 'numpy_argmax']:
            assert len(args) in [1, 2], f"numpy_<reduce> should either one or two arguments, but got {len(args)}"
            func_args = [self.node_shapes[args[0]]]
            if len(args) == 2:
                if not (isinstance(args[1], ast.Constant) and isinstance(args[1].value, int)):
                    raise RuntimeError("Second argument for numpy_<reduce> should be an int constant for shape analysis")
                
                func_args.append(args[1].value)
            return func_table.numpy_reduce_generic(*func_args)
        else:
            f = getattr(func_table, f_name)
            return f(*[self.node_shapes[arg] for arg in args])
        
    def visit_Call(self, node: ast.Call):
        for arg in node.args:
            self.visit(arg)

        if not (
            (isinstance(node.func, ast.Name) or 
             isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name))
        ):
            raise RuntimeError("Function calls only suport named calls or named module calls")

        if isinstance(node.func, ast.Name):
            f_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            if not node.func.value.id in self.modules:
                raise KeyError(f"Module {node.func.value.id} not found in runtime vals")
            
            module_name = self.modules[node.func.value.id].__name__
            f_name = f"{module_name}_{node.func.attr}"
        else:
            assert False, "Impossible path"

        self.node_shapes[node] = self.dispatch_call(f_name, node.args)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        indices = []
        if isinstance(node.slice, ast.Tuple):
            for index in node.slice.elts:
                indices.append(self.node_shapes[index])
        else:
            indices.append(self.node_shapes[node.slice])
        f = getattr(func_table, 'subscript')
        self.node_shapes[node] = f(self.node_shapes[node.value], indices)

    def visit_Slice(self, node: ast.Slice):  
        if (
            isinstance(node.upper, ast.UnaryOp)
            and isinstance(node.upper.op, ast.USub) 
            and isinstance(node.upper.operand, ast.Constant)
        ):
            node.upper = ast.Constant(value=-node.upper.operand.value)

        args = []
        for arg in [node.lower, node.upper, node.step]:
            if arg is None:
                args.append(None)
            elif isinstance(arg, ast.Constant):
                args.append(arg.value)
            elif isinstance(arg, ast.expr):
                args.append(ast.unparse(arg))
            else:
                raise RuntimeError("Should not reach here")
        
        f = getattr(func_table, 'slice')
        self.node_shapes[node] = f(*args)

class AnalyzeAssignShapes(AnalyzeExprShapes):
    def __init__(self, rt_vals):
        super().__init__(rt_vals)

    def visit_Assign(self, node):
        self.visit(node.value)
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id not in self.var_shapes:
            self.var_shapes[target.id] = self.node_shapes[node.value]

        self.visit(target)
        # Check if the shape of the target and the value are the same
        if self.node_shapes[target] != self.node_shapes[node.value]:
            raise RuntimeError(f"Shapes mismatch for assignment: {ast.unparse(node)}")
        
    def visit_For(self, node):
        # Only for-range calls are supported
        assert is_call(node.iter, ["range"])
        self.var_shapes[node.target.id] = ()
        self.generic_visit(node)


def analyze(tree, rt_vals):
    visitor = AnalyzeAssignShapes(rt_vals)
    visitor.visit(tree)
    return visitor.node_shapes