from __future__ import annotations

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from findex.index import Index
from findex.tokenize import tokenize


def _adjacent_positions(
    pos_a: tuple[int, ...],
    pos_b: tuple[int, ...],
    offset: int,
) -> bool:
    """Two-pointer check: exists p in pos_a such that (p + offset) in pos_b."""
    i = 0
    j = 0
    while i < len(pos_a) and j < len(pos_b):
        target = pos_a[i] + offset
        if pos_b[j] == target:
            return True
        if pos_b[j] < target:
            j += 1
        else:
            i += 1
    return False


class QueryNode(ABC):
    """Base node for composable Boolean and phrase query ASTs."""

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    @abstractmethod
    def evaluate(self, index: Index) -> set[int]:
        """Return matching doc_ids in `index`."""


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        if self.value not in index:
            return set()
        return {p.doc_id for p in index[self.value]}


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...] | list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()
        if len(self.terms) == 1:
            return Term(self.terms[0]).evaluate(index)
        if not index.include_positions:
            raise ValueError(
                "Phrase queries require positional postings. Rebuild index with --positions."
            )

        posting_lists = [index[t] if t in index else [] for t in self.terms]
        if any(not pl for pl in posting_lists):
            return set()

        candidate_docs = set.intersection(
            *({p.doc_id for p in pl} for pl in posting_lists)
        )
        if not candidate_docs:
            return set()

        pos_maps = [
            {p.doc_id: p.positions for p in pl if p.doc_id in candidate_docs}
            for pl in posting_lists
        ]

        matched: set[int] = set()
        for doc_id in candidate_docs:
            first_positions = pos_maps[0][doc_id]
            if all(
                _adjacent_positions(first_positions, pos_maps[offset][doc_id], offset)
                for offset in range(1, len(self.terms))
            ):
                matched.add(doc_id)
        return matched


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        left_docs = self.left.evaluate(index)
        if not left_docs:
            return set()
        return left_docs & self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        universe = set(index.doc_meta.keys())
        return universe - self.operand.evaluate(index)


def positive_terms(node: QueryNode | None) -> list[str]:
    """Collect unique positive query terms (excluding NOT subtrees) in order."""
    if node is None:
        return []
    seen: list[str] = []

    def _walk(cur: QueryNode) -> None:
        if isinstance(cur, Term):
            if cur.value not in seen:
                seen.append(cur.value)
        elif isinstance(cur, Phrase):
            for t in cur.terms:
                if t not in seen:
                    seen.append(t)
        elif isinstance(cur, (And, Or)):
            _walk(cur.left)
            _walk(cur.right)
        elif isinstance(cur, Not):
            return

    _walk(node)
    return seen


_LEX_RE = re.compile(r'\s*(?:(\()|(\))|"([^"]*)"|([^\s()"]+))')


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> tuple[str, str] | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def parse(self) -> QueryNode | None:
        if not self._tokens:
            return None
        node = self._parse_or()
        if self._peek() is not None:
            raise SyntaxError(f"Unexpected token in query: {self._peek()!r}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self._peek()) is not None and tok == ("OP", "OR"):
            self._consume()
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self._peek()) is not None:
            if tok == ("RPAREN", ")") or tok == ("OP", "OR"):
                break
            if tok == ("OP", "AND"):
                self._consume()
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> QueryNode:
        tok = self._peek()
        if tok == ("OP", "NOT"):
            self._consume()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> QueryNode:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of query")

        kind, val = self._consume()
        if kind == "LPAREN":
            expr = self._parse_or()
            if self._peek() != ("RPAREN", ")"):
                raise SyntaxError("Unclosed '(' in query")
            self._consume()
            return expr

        if kind == "PHRASE":
            norm_tokens = tuple(tokenize(val))
            if not norm_tokens:
                raise SyntaxError("Empty phrase in query")
            if len(norm_tokens) == 1:
                return Term(norm_tokens[0])
            return Phrase(norm_tokens)

        if kind == "WORD":
            norm_tokens = list(tokenize(val))
            if not norm_tokens:
                raise SyntaxError(f"Invalid search token: {val!r}")
            node: QueryNode = Term(norm_tokens[0])
            for extra in norm_tokens[1:]:
                node = And(node, Term(extra))
            return node

        raise SyntaxError(f"Unexpected token {val!r} in query")


def parse(query: str) -> QueryNode | None:
    """Parse a query string into a QueryNode AST."""
    if not query.strip():
        return None
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(query):
        match = _LEX_RE.match(query, pos)
        if not match:
            raise SyntaxError(
                f"Invalid query syntax near position {pos}: {query[pos:]!r}"
            )
        pos = match.end()
        lparen, rparen, phrase, word = match.groups()
        if lparen is not None:
            tokens.append(("LPAREN", "("))
        elif rparen is not None:
            tokens.append(("RPAREN", ")"))
        elif phrase is not None:
            tokens.append(("PHRASE", phrase))
        elif word is not None:
            if word in {"AND", "OR", "NOT"}:
                tokens.append(("OP", word))
            else:
                tokens.append(("WORD", word))

    return _Parser(tokens).parse()