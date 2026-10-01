"""Turn a cached ontology entity into a compact, token-budgeted concept description.

Representation levels (the paper's ablation axis):
  A  label only
  B  label + synonyms
  C  label + synonyms + definition
  D  label + synonyms + definition + one-hop parent labels          (default)

LAYA truncates an over-long state from the right, which would silently drop the TARGET concept.
So each concept is fitted to its own token budget here, with a fixed priority: the label is
always kept whole, then synonyms and parents are added while they fit their share, and the
definition is cut at a word boundary to whatever remains.
"""
from __future__ import annotations

REPRESENTATIONS = ("A", "B", "C", "D")
# share of the per-concept token budget a field may use (definition takes what is left)
SYNONYM_SHARE = 0.35
PARENT_SHARE = 0.15


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _fit_list(tok, prefix: str, items: list[str], budget: int, sep: str = "; ") -> str:
    """Longest prefix of `items` whose rendered line fits `budget` tokens ('' if none fit)."""
    line, kept = "", []
    for item in items:
        candidate = prefix + sep.join(kept + [item])
        if n_tokens(tok, candidate) > budget:
            break
        kept.append(item)
        line = candidate
    return line


def _fit_text(tok, prefix: str, text: str, budget: int) -> str:
    """`prefix + text` cut at a word boundary to fit `budget` tokens ('' if nothing fits)."""
    full = prefix + text
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) <= budget:
        return full
    if budget <= n_tokens(tok, prefix) + 4:
        return ""
    end = enc["offset_mapping"][budget - 2][1]  # leave room for the ellipsis
    cut = full[:end].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + " ..." if len(cut) > len(prefix) else ""


def entity_text(entity: dict, ontology: str, role: str, rep: str, tok, max_tokens: int) -> str:
    """Render one concept. `role` is 'SOURCE' or 'TARGET'."""
    if rep not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {rep!r}")
    lines = [f"{role} CONCEPT", f"Ontology: {ontology}", f"Preferred label: {entity['label']}"]
    used = n_tokens(tok, "\n".join(lines))
    remaining = max(0, max_tokens - used)

    if rep in ("B", "C", "D"):
        names = entity["synonyms"] + entity["related_synonyms"]
        share = remaining if rep == "B" else int(max_tokens * SYNONYM_SHARE)
        line = _fit_list(tok, "Synonyms: ", names, min(share, remaining)) if names else ""
        if line:
            lines.append(line)
            remaining -= n_tokens(tok, "\n" + line)

    parent_line = ""
    if rep == "D" and entity["parents"]:
        parent_line = _fit_list(tok, "Parent concepts: ", entity["parents"],
                                min(int(max_tokens * PARENT_SHARE), remaining))
        if parent_line:
            remaining -= n_tokens(tok, "\n" + parent_line)

    if rep in ("C", "D") and entity["definition"]:
        line = _fit_text(tok, "Definition: ", entity["definition"], remaining - 1)
        if line:
            lines.append(line)
    if parent_line:
        lines.append(parent_line)
    return "\n".join(lines)
