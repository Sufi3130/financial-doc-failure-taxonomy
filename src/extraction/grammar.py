"""GBNF grammar: constrain decoding to the Receipt JSON schema (issue #25, opt-in).

Off by default in the pipeline. Forcing valid JSON would hide the
``malformed_json`` failures the taxonomy measures, so this is a *selectable*
mitigation (``--grammar`` / a UI toggle), not the baseline. When on, llama.cpp
can only emit exactly one object with the four schema keys in order, each a JSON
string or ``null`` -- prose and broken JSON become unreachable.

    from src.extraction.grammar import load_grammar, all_null_json
    grammar = load_grammar()            # a llama_cpp.LlamaGrammar
    ... create_chat_completion(..., grammar=grammar)

The grammar is built from ``schema.FIELDS`` so it cannot drift from the schema.
"""

import json

from .schema import FIELDS


def grammar_gbnf():
    """The GBNF text forcing {"<field>": string|null, ...} for exactly FIELDS, in order."""
    pairs = [f'"\\"{f}\\"" ws ":" ws value' for f in FIELDS]
    body = ' ws "," ws '.join(pairs)
    return (
        f'root ::= "{{" ws {body} ws "}}"\n'
        'value ::= string | "null"\n'
        'string ::= "\\"" char* "\\""\n'
        r'char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)' + "\n"
        'hex ::= [0-9a-fA-F]\n'
        'ws ::= [ \\t\\n]*\n'
    )


def load_grammar():
    """Compile the grammar into a llama_cpp.LlamaGrammar (imports llama_cpp lazily)."""
    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_string(grammar_gbnf(), verbose=False)


def all_null_json():
    """The deterministic all-null answer for an empty/unreadable page."""
    return json.dumps({f: None for f in FIELDS})
