from hypothesis import given, strategies as st
import pytest

from findex.index import Index
from findex.query import And, Not, Or, Phrase, QueryNode, Term, explain_query, format_query, parse
from findex.search import boolean_search


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("python", Term("python")),
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("(a OR b) c", And(Or(Term("a"), Term("b")), Term("c"))),
        ("NOT java", Not(Term("java"))),
        ('"event loop"', Phrase(("event", "loop"))),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_tree_equality(raw: str, expected: QueryNode | None) -> None:
    assert parse(raw) == expected


def test_operator_overloading_and_explain() -> None:
    ast = (Term("python") | Term("rust")) & ~Term("java")
    assert ast == And(Or(Term("python"), Term("rust")), Not(Term("java")))
    rendered = explain_query(ast)
    assert "AND" in rendered and "OR" in rendered and "NOT" in rendered


@pytest.mark.parametrize("bad_query", ['"unclosed phrase', "(python OR rust", "python )"])
def test_parse_syntax_errors(bad_query: str) -> None:
    with pytest.raises(SyntaxError):
        parse(bad_query)


def test_boolean_and_phrase_evaluation(sample_index: Index) -> None:
    assert boolean_search(sample_index, "python") == [0, 3]
    assert boolean_search(sample_index, "python AND lazy") == [0]
    assert boolean_search(sample_index, '"event loop"') == [3]
    assert boolean_search(sample_index, '"loop event"') == []
    assert boolean_search(sample_index, "python NOT lazy") == [3]
    assert boolean_search(sample_index, "") == []


_WORD_STRATEGY = st.from_regex(r"[a-z]{2,8}", fullmatch=True)


def _query_ast_strategy() -> st.SearchStrategy[QueryNode]:
    leaves: st.SearchStrategy[QueryNode] = st.one_of(
        _WORD_STRATEGY.map(Term),
        st.lists(_WORD_STRATEGY, min_size=2, max_size=3).map(Phrase),
    )
    return st.recursive(
        leaves,
        lambda children: st.one_of(
            st.tuples(children, children).map(lambda pair: And(pair[0], pair[1])),
            st.tuples(children, children).map(lambda pair: Or(pair[0], pair[1])),
            children.map(Not),
        ),
        max_leaves=6,
    )


@given(_query_ast_strategy())
def test_property_query_format_parse_roundtrip(ast: QueryNode) -> None:
    rendered = format_query(ast)
    reparsed = parse(rendered)
    assert reparsed == ast


@given(_WORD_STRATEGY, _WORD_STRATEGY)
def test_property_boolean_oracle_matches_python_sets(
    sample_index: Index, t1: str, t2: str
) -> None:
    s1 = {p.doc_id for p in sample_index.get(t1, ())}
    s2 = {p.doc_id for p in sample_index.get(t2, ())}
    assert And(Term(t1), Term(t2)).evaluate(sample_index) == (s1 & s2)
    assert Or(Term(t1), Term(t2)).evaluate(sample_index) == (s1 | s2)