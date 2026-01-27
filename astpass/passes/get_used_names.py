import ast

class GetUsedNames(ast.NodeVisitor):
    def __init__(self, no_funcname):
        self.used = []
        self.no_funcname = no_funcname

    def visit_Name(self, node):
        if node.id not in self.used:
            self.used.append(node.id)

    def visit_Call(self, node: ast.Call):
        if self.no_funcname:
            for arg in node.args:
                self.visit(arg)
        else:
            self.generic_visit(node)

def analyze(tree, no_funcname):
    visitor = GetUsedNames(no_funcname)
    visitor.visit(tree)
    return visitor.used