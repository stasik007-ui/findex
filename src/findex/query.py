"""Public query-language API."""

from findex.ast import And, Not, Or, Phrase, QueryNode, Term, parse, positive_terms


def format_query(node: QueryNode) -> str:
    """Render a query tree so that parsing it recreates the same tree."""
    if isinstance(node, Term):
        return node.value
    if isinstance(node, Phrase):
        return '"' + " ".join(node.terms) + '"'
    if isinstance(node, Not):
        return f"NOT ({format_query(node.operand)})"
    if isinstance(node, And):
        return f"({format_query(node.left)} AND {format_query(node.right)})"
    if isinstance(node, Or):
        return f"({format_query(node.left)} OR {format_query(node.right)})"
    raise TypeError(f"Unsupported query node: {type(node).__name__}")


def explain_query(node: QueryNode) -> str:
    """Return a readable representation of a parsed query tree."""
    if isinstance(node, Term):
        return node.value
    if isinstance(node, Phrase):
        return '"' + " ".join(node.terms) + '"'
    if isinstance(node, Not):
        return f"NOT {explain_query(node.operand)}"
    if isinstance(node, And):
        return f"({explain_query(node.left)} AND {explain_query(node.right)})"
    if isinstance(node, Or):
        return f"({explain_query(node.left)} OR {explain_query(node.right)})"
    raise TypeError(f"Unsupported query node: {type(node).__name__}")


__all__ = [
    "And",
    "Not",
    "Or",
    "Phrase",
    "QueryNode",
    "Term",
    "explain_query",
    "format_query",
    "parse",
    "positive_terms",
]
