#!/usr/bin/env python3
"""
SY annotation: the 200 evaluation sequences, decoded and labelled for the
groupings of protocol §5.22 (design note §6), aligned to GPT-2 BPE tokens.

Runs in the annotation environment, not the model environment: spaCy and
fastcoref need numpy and transformers versions that the torch 2.2.2 model
stack cannot share, so they live in their own virtualenv and this script
writes a cache the model scripts read.

    ~/.venvs/sy-annotate/bin/python sy_annotate.py

Input   sy_cache/sy_tokens.npz       (x: 200 x 512 token ids, from the SY token dump)
Output  sy_cache/sy_annotation.npz   per-token labels, -1 where a label does not apply:
          word      index of the spaCy word the token belongs to (-1: whitespace only)
          pos       coarse part of speech of that word (index into POS_TAGS)
          head      index of the word's syntactic head (its own index at the root)
          punct     1 if the word is punctuation
          chunk     noun-chunk id within the sequence
          ent       named-entity span id within the sequence
          ent_type  entity label (index into ENT_TAGS)
          coref     coreference cluster id within the sequence
          mention   mention id within the sequence (pairs within one mention are excluded)
        sy_cache/sy_annotation_text.json  the decoded texts and per-token character spans

Alignment: each token's bytes go through an incremental UTF-8 decoder, so the
concatenated outputs are exactly the decoded text and a character split across
two tokens belongs to the token that completes it. A token takes the label of
the word containing its first non-space character.
"""
import codecs, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "sy_cache"
POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
            "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
ENT_TAGS = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP",
            "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]


def decode_with_spans(ids, tok):
    """Exact decoded text and the [start, end) character span of every token."""
    byte_dec = tok.byte_decoder if hasattr(tok, "byte_decoder") else None
    if byte_dec is None:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_dec = {v: k for k, v in bytes_to_unicode().items()}
    pieces = tok.convert_ids_to_tokens([int(i) for i in ids])
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text, spans = [], []
    n = 0
    for p in pieces:
        out = dec.decode(bytes(byte_dec[c] for c in p))
        spans.append((n, n + len(out)))
        text.append(out); n += len(out)
    tail = dec.decode(b"", final=True)
    text.append(tail)
    return "".join(text), spans


def main():
    import spacy
    from transformers import AutoTokenizer
    from fastcoref import FCoref

    x = np.load(CACHE / "sy_tokens.npz")["x"]
    N, T = x.shape
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    texts, spans = [], []
    for i in range(N):
        t, s = decode_with_spans(x[i], tok)
        texts.append(t); spans.append(s)
    # the decode is exact: re-encoding the decoded text gives back the tokens wherever no character was split
    exact = sum(tok.decode([int(v) for v in x[i]], clean_up_tokenization_spaces=False) == texts[i] for i in range(N))
    print(f"decoded {N} sequences; text identical to the tokenizer's own decode in {exact}/{N}", flush=True)

    t0 = time.time()
    nlp = spacy.load("en_core_web_trf")
    docs = list(nlp.pipe(texts, batch_size=8))
    print(f"spaCy en_core_web_trf: {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    coref = FCoref(device="cpu")
    preds = coref.predict(texts=texts)
    print(f"fastcoref FCoref: {time.time() - t0:.0f} s", flush=True)

    L = {k: -np.ones((N, T), np.int32) for k in ("word", "pos", "head", "punct", "chunk", "ent", "ent_type", "coref", "mention")}
    stats = {"tokens_with_word": 0, "noun_chunks": 0, "entities": 0, "coref_clusters": 0, "coref_mentions": 0}
    for i, (doc, sp, pr) in enumerate(zip(docs, spans, preds)):
        text = texts[i]
        char_word = -np.ones(len(text) + 1, np.int32)
        for w in doc:
            char_word[w.idx:w.idx + len(w.text)] = w.i
        char_chunk = -np.ones(len(text) + 1, np.int32)
        for c, ch in enumerate(doc.noun_chunks):
            char_chunk[ch.start_char:ch.end_char] = c
        char_ent = -np.ones(len(text) + 1, np.int32); char_ety = -np.ones(len(text) + 1, np.int32)
        for e, ent in enumerate(doc.ents):
            char_ent[ent.start_char:ent.end_char] = e
            char_ety[ent.start_char:ent.end_char] = ENT_TAGS.index(ent.label_) if ent.label_ in ENT_TAGS else -1
        char_coref = -np.ones(len(text) + 1, np.int32); char_mention = -np.ones(len(text) + 1, np.int32)
        m_id = 0
        clusters = pr.get_clusters(as_strings=False)
        for c, cl in enumerate(clusters):
            for (a, b) in cl:
                char_coref[a:b] = c; char_mention[a:b] = m_id; m_id += 1
        stats["noun_chunks"] += len(list(doc.noun_chunks)); stats["entities"] += len(doc.ents)
        stats["coref_clusters"] += len(clusters); stats["coref_mentions"] += m_id
        for t, (a, b) in enumerate(sp):
            seg = text[a:b]
            k = len(seg) - len(seg.lstrip())
            if a + k >= b:
                continue                                  # whitespace-only token
            c0 = a + k
            w = char_word[c0]
            if w < 0:
                continue
            wd = doc[int(w)]
            L["word"][i, t] = w
            L["pos"][i, t] = POS_TAGS.index(wd.pos_) if wd.pos_ in POS_TAGS else -1
            L["head"][i, t] = wd.head.i
            L["punct"][i, t] = int(wd.is_punct or wd.pos_ == "PUNCT")
            L["chunk"][i, t] = char_chunk[c0]
            L["ent"][i, t] = char_ent[c0]; L["ent_type"][i, t] = char_ety[c0]
            L["coref"][i, t] = char_coref[c0]; L["mention"][i, t] = char_mention[c0]
            stats["tokens_with_word"] += 1
    np.savez_compressed(CACHE / "sy_annotation.npz", **L, pos_tags=np.array(POS_TAGS), ent_tags=np.array(ENT_TAGS))
    (CACHE / "sy_annotation_text.json").write_text(json.dumps({"texts": texts, "spans": spans}))
    cov = {k: float((L[k] >= 0).mean()) for k in ("word", "chunk", "ent", "coref")}
    print("labelled share of tokens: " + ", ".join(f"{k} {v:.3f}" for k, v in cov.items()))
    print("counts over 200 sequences: " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    print(f"wrote {CACHE / 'sy_annotation.npz'}")


if __name__ == "__main__":
    main()
