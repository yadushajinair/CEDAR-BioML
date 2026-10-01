"""The shared LAYA equivalence question and the (source, target) pair states it is asked about.

One binary `choice` question is reused for every pair; only the state changes. This is what
makes `Agent.predict_batch` the right throughput path.
"""
from __future__ import annotations

import os

from build_entity_text import entity_text, n_tokens
from ontology_loader import load_entities
from utils import PAIRS

QID = "equivalence"
EQUIVALENT = "EQUIVALENT"
NOT_EQUIVALENT = "NOT_EQUIVALENT"

# Neutral option labels (not true/false): LAYA's `noul` primitive is documented to follow its
# false/true labels instead of the state on the English checkpoint (model card, issue #156).
QUESTIONS = {
    QID: {
        "type": "choice",
        "instructions": (
            "Determine whether the source ontology class and target ontology class denote the "
            "same biomedical concept. Equivalence means the same concept, not merely related, "
            "broader, narrower, parent, child, or frequently associated."
        ),
        "criteria": {
            EQUIVALENT: (
                "The two classes denote the same biomedical concept and could represent an "
                "ontology equivalence correspondence."
            ),
            NOT_EQUIVALENT: (
                "The two classes do not denote the same concept; this includes broader, "
                "narrower, related, associated, or unrelated concepts."
            ),
        },
    }
}

LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"  # English checkpoint, repo root


def load_tokenizer(model: str = LAYA_REPO):
    """The tokenizer of a LAYA checkpoint (Hub id pinned to LAYA_REVISION, or a local dir)."""
    from transformers import AutoTokenizer

    if os.path.isdir(model):
        return AutoTokenizer.from_pretrained(os.path.join(model, "tokenizer"))
    from huggingface_hub import snapshot_download

    path = snapshot_download(model, revision=LAYA_REVISION, allow_patterns=[
        "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    return AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))


def head_tokens(tok, questions: dict = QUESTIONS) -> int:
    """Tokens LAYA spends on `[CLS] head [SEP] [MASK] opt ... [SEP]` for the question, measured
    with LAYA's own sequence builder on an empty state."""
    from laya.common import build_sequence

    q = questions[QID]
    seq, _markers = build_sequence(tok, "", {"t": q["type"], "ins": q["instructions"],
                                             "crit": q["criteria"]}, 4096, 192)
    return len(seq) - 1  # minus the closing [SEP] that follows the (empty) state


class PairStateBuilder:
    """Builds and caches concept descriptions, then joins them into pair states."""

    def __init__(self, pair: str, rep: str, tok, max_len: int = 512):
        info = PAIRS[pair]
        self.pair, self.rep, self.tok = pair, rep, tok
        self.src_onto, self.tgt_onto = info["src"], info["tgt"]
        self.src_entities = load_entities(self.src_onto)
        self.tgt_entities = load_entities(self.tgt_onto)
        # state room = max_len - head - closing [SEP]; keep a small safety margin for the
        # separator between the two concepts and tokenizer merge effects at the joins.
        self.state_room = max_len - head_tokens(tok) - 1
        self.per_concept = (self.state_room - 8) // 2
        self._cache: dict[tuple[str, str], str] = {}

    def concept(self, iri: str, role: str) -> str:
        key = (role, iri)
        if key not in self._cache:
            onto = self.src_onto if role == "SOURCE" else self.tgt_onto
            ents = self.src_entities if role == "SOURCE" else self.tgt_entities
            if iri not in ents:
                raise KeyError(f"{iri} is not in the {onto} entity cache")
            self._cache[key] = entity_text(ents[iri], onto, role, self.rep, self.tok,
                                           self.per_concept)
        return self._cache[key]

    def state(self, src: str, tgt: str) -> str:
        return self.concept(src, "SOURCE") + "\n\n" + self.concept(tgt, "TARGET")

    def state_tokens(self, src: str, tgt: str) -> int:
        return n_tokens(self.tok, self.state(src, tgt))
