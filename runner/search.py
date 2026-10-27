"""Identify repeated programs without executing them."""

import ast
from runner.evaluate import digest


def structural_hash(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is not None:
                node.body.pop(0)
    return digest(ast.dump(tree, include_attributes=False))
