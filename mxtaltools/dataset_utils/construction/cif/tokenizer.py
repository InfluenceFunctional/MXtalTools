"""CIF 1.1 tokenizer: text -> data blocks of tags and loops.

Deliberately LOSSLESS and chemistry-free.  It returns strings exactly as written;
interpreting them (stripping estimated standard deviations, mapping `.`/`?` to
missing, coercing to float) is the caller's job via the helpers at the bottom.
Keeping those separate means a parse bug and an interpretation bug cannot be
confused for each other.

Scope is the subset CSD exports and this package's own writer actually use:
`data_` blocks, `loop_`, single/double quoted values, `;`-delimited multi-line
text, and `#` comments.  Not supported, and refused loudly rather than guessed:
save frames (`save_`), which do not occur in either source.
"""

from typing import Dict, List, Optional, Tuple, Union

from .errors import CifParseError

__all__ = ['CifBlock', 'CifLoop', 'parse_cif', 'strip_esd', 'is_missing', 'as_float', 'as_int']

MISSING = ('.', '?')


class CifLoop:
    """One `loop_`: an ordered tag list and the rows under it."""

    __slots__ = ('tags', 'rows')

    def __init__(self, tags: List[str], rows: List[List[str]]):
        self.tags = tags
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, tag: str) -> bool:
        return tag.lower() in [t.lower() for t in self.tags]

    def column(self, tag: str) -> List[str]:
        """Every value under `tag`, in row order. Case-insensitive, as CIF requires."""
        lower = [t.lower() for t in self.tags]
        try:
            i = lower.index(tag.lower())
        except ValueError:
            raise KeyError(f'loop has no tag {tag!r}; it has {self.tags}')
        return [r[i] for r in self.rows]

    def __repr__(self) -> str:
        return f'CifLoop({len(self.tags)} tags x {len(self.rows)} rows)'


class CifBlock:
    """One `data_` block: scalar tag->value pairs plus its loops."""

    __slots__ = ('name', 'tags', 'loops')

    def __init__(self, name: str):
        self.name = name
        self.tags: Dict[str, str] = {}
        self.loops: List[CifLoop] = []

    def get(self, tag: str, default=None) -> Optional[str]:
        """Scalar lookup, case-insensitive. Returns `default` if absent."""
        v = self.tags.get(tag.lower())
        return default if v is None else v

    def loop_with(self, tag: str) -> Optional[CifLoop]:
        """The first loop carrying `tag`, or None."""
        for loop in self.loops:
            if tag in loop:
                return loop
        return None

    def first_of(self, *tags: str) -> Tuple[Optional[str], Optional[str]]:
        """The first present tag among `tags`, as (tag, value).

        For dialect pairs: CSD exports carry the deprecated `_symmetry_*`
        spellings while this package's writer emits the current `_space_group_*`
        ones.  Callers must accept both, and knowing WHICH matched is sometimes
        diagnostic, so the tag is returned alongside the value.
        """
        for t in tags:
            v = self.get(t)
            if v is not None:
                return t, v
        return None, None

    def __repr__(self) -> str:
        return f'CifBlock({self.name!r}, {len(self.tags)} tags, {len(self.loops)} loops)'


def _tokenize(text: str):
    """Yield (kind, value) where kind is 'tag', 'value', 'loop' or 'data'.

    Handles `;`-delimited text fields, which may contain anything including
    what look like tags, so they must be consumed before any other rule.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith(';'):                     # multi-line text field
            buf = [stripped[1:]]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(';'):
                buf.append(lines[i])
                i += 1
            if i >= len(lines):
                raise CifParseError('unterminated ";" text field')
            i += 1
            yield 'value', '\n'.join(buf).strip()
            continue

        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        for token in _split_line(stripped):
            low = token.lower()
            if low.startswith('data_'):
                yield 'data', token[5:]
            elif low == 'loop_':
                yield 'loop', None
            elif low.startswith('save_'):
                raise CifParseError('save frames are not supported')
            elif token.startswith('_'):
                yield 'tag', token
            else:
                yield 'value', token
        i += 1


def _split_line(line: str) -> List[str]:
    """Split one line into tokens, honouring quotes and stripping trailing comments.

    A `#` only starts a comment outside quotes and at a token boundary -- inside a
    value it is legal, and CSD identifiers do contain odd characters.
    """
    out, buf, quote = [], [], None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            # a quote only closes at whitespace or end-of-line, per CIF 1.1
            if ch == quote and (i + 1 == len(line) or line[i + 1].isspace()):
                out.append(''.join(buf)); buf = []; quote = None
            else:
                buf.append(ch)
        elif ch in ("'", '"') and not buf:
            quote = ch
        elif ch == '#' and not buf:
            break
        elif ch.isspace():
            if buf:
                out.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
        i += 1
    if quote:
        raise CifParseError(f'unterminated {quote} quote in line: {line[:80]!r}')
    if buf:
        out.append(''.join(buf))
    return out


def parse_cif(text: str) -> List[CifBlock]:
    """Parse CIF text into its data blocks, in file order.

    Raises `CifParseError` on a malformed loop (tags with no rows, or a row count
    that is not a multiple of the tag count) rather than silently truncating --
    a ragged loop that is quietly padded is how an atom site acquires the wrong
    element.
    """
    blocks: List[CifBlock] = []
    block: Optional[CifBlock] = None
    pending_tag: Optional[str] = None
    loop_tags: Optional[List[str]] = None
    loop_values: Optional[List[str]] = None
    in_loop_header = False

    def close_loop():
        nonlocal loop_tags, loop_values, in_loop_header
        if loop_tags is None:
            return
        if not loop_tags:
            raise CifParseError('loop_ with no tags')
        n = len(loop_tags)
        if len(loop_values) % n:
            raise CifParseError(
                f'ragged loop: {len(loop_values)} values is not a multiple of '
                f'{n} tags ({loop_tags[:3]}...)')
        rows = [loop_values[k:k + n] for k in range(0, len(loop_values), n)]
        block.loops.append(CifLoop(loop_tags, rows))
        loop_tags = loop_values = None
        in_loop_header = False

    for kind, value in _tokenize(text):
        if kind == 'data':
            close_loop()
            block = CifBlock(value)
            blocks.append(block)
            pending_tag = None
            continue

        if block is None:
            raise CifParseError(f'{kind} {value!r} appears before any data_ block')

        if kind == 'loop':
            close_loop()
            loop_tags, loop_values, in_loop_header = [], [], True
        elif kind == 'tag':
            if in_loop_header:
                loop_tags.append(value)
            else:
                close_loop()
                pending_tag = value
        else:                                             # value
            if loop_tags is not None:
                in_loop_header = False
                loop_values.append(value)
            elif pending_tag is not None:
                block.tags[pending_tag.lower()] = value
                pending_tag = None
            else:
                raise CifParseError(f'value {value!r} with no preceding tag')

    close_loop()
    if not blocks:
        raise CifParseError('no data_ block found')
    return blocks


# --- interpretation helpers, kept separate from parsing ----------------------

def is_missing(value: Optional[str]) -> bool:
    """CIF spells missing as `.` (inapplicable) or `?` (unknown)."""
    return value is None or value.strip() in MISSING


def strip_esd(value: str) -> str:
    """`'1.234(5)'` -> `'1.234'`. The parenthesised standard deviation is dropped.

    Returned as a string so the caller decides int vs float.
    """
    v = value.strip()
    cut = v.find('(')
    return v[:cut].strip() if cut != -1 else v


def as_float(value: Optional[str], tag: str = '', identifier: str = None) -> Optional[float]:
    """Parse a CIF numeric to float, or None if missing. Raises on garbage."""
    if is_missing(value):
        return None
    try:
        return float(strip_esd(value))
    except ValueError:
        raise CifParseError(f'{tag or "value"} {value!r} is not numeric', identifier)


def as_int(value: Optional[str], tag: str = '', identifier: str = None) -> Optional[int]:
    """Parse a CIF numeric to int, or None if missing. Raises on garbage.

    Accepts `'4'` and `'4.0'` -- CSD writes `_cell_formula_units_Z` both ways --
    but refuses a genuinely fractional value, which would mean the tag does not
    hold what the caller thinks.
    """
    f = as_float(value, tag, identifier)
    if f is None:
        return None
    if abs(f - round(f)) > 1e-8:
        raise CifParseError(f'{tag or "value"} {value!r} is not an integer', identifier)
    return int(round(f))
