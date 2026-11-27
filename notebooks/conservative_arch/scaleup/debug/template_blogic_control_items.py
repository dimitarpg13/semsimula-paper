"""Premises of the B-logic control (2026-11-27; BLogic_Test.md §5.4): levels BCa and BCb, contexts that name both clauses.
Same frames, wordings, kinds and pairs as template_blogic_items.py; for template_analogy_test.py (bank=blogic_control) and the scorers
(items=template_blogic_control_items)."""
import template_blogic_items as _B

LEVEL_ORDER = ('BCa', 'BCb'); PAIRS = {lv: _B.PAIRS[lv] for lv in LEVEL_ORDER}


def build(enc, val=None):
    return _B.build(enc, val, levels=LEVEL_ORDER)
