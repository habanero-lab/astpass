import ast

def is_call(node, names=None):
    '''
    Check if the node is a call node. If `names` is not None, check if the function name is in `names`.
    '''
    if isinstance(node, ast.Call):
        if names:
            names = names if isinstance(names, (tuple, list)) else (names,)
            return ast.unparse(node.func) in names
        else:
            return True
    else:
        return False
    
def str_to_ast_expr(expr_str):
    return ast.parse(expr_str).body[0].value
