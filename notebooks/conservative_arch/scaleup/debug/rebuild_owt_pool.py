"""Rebuild the ladder's shared OpenWebText pool on a Colab CPU runtime (2026-11-12).

    !pip -q install "datasets<4"          # the dataset is loaded through its script, as in July
    !python rebuild_owt_pool.py [out=/content/drive/MyDrive/semsimula_owt_gpt2/data]

The pool every Gen 3 OpenWebText arm trained on (openwebtext_train_2000M.npy + openwebtext_val_2M.npy) lived in
one Drive folder, which was deleted on 2026-11-12. This repeats the procedure that made it, exactly as the
gamma-sweep notebooks of July and the ladder's Cell 3 still write it: stream Skylion007/openwebtext in order, join
50,000 documents at a time with a blank line, GPT-2-tokenise each joined chunk, stop once 2,002,000,000 tokens are
in hand; validation is the last 2,000,000 tokens of that stream and training the first 2,000,000,000. The only
change is memory and speed, with the same tokens: the tokens go into one preallocated uint16 array instead of a
Python list of ints (about 70 GB), and each joined chunk (about 250 MB of text) is tokenised in pieces of about
1 MB, cut where a non-whitespace character is followed by a space or newline. GPT-2's pre-tokenizer never forms a
piece across such a point, so the pieces' tokens concatenate to exactly the joined chunk's (checked 2026-11-12 on
11M characters of mixed text and whitespace edge cases, at piece sizes from 1 kB to 1 MB: identical); the batch
call runs on all cores. Peak memory is about 7 GB, so a standard CPU runtime suffices (High-RAM is safer).
Special tokens are not added (`add_special_tokens=False`), as GPT-2's `encode` did not add any in July.

The proof that the stream reproduced is the validation file: it is the tail of the stream, so it matches the
original only if every chunk before it did. The original's fingerprint is that of the validation file the CPU
harness reads (gradcheck_vphi_xi_paths.VAL), which reproduced a Colab Cell 6b-7 evaluation exactly (58.02,
protocol, D3 damping dial). Validation is written and checked before the 4 GB training file.
"""
import hashlib
import os
import sys
import time

import numpy as np

OPTS = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
OUT = OPTS.get('out', '/content/drive/MyDrive/semsimula_owt_gpt2/data')
MAX_TRAIN_TOKENS = 2_000_000_000
VAL_TOKENS = 2_000_000
CHUNK_SIZE = 50_000
VAL_FILE_MD5 = 'a4e8b4afd25cebe671dbd99dcbf4ad01'      # md5 of the original openwebtext_val_2M.npy file
VAL_TOKENS_MD5 = '132035c0a57c50bb71f7f5d2944910d6'    # md5 of its tokens (ids.tobytes()), header-independent
VAL_HEAD = [7866, 4556, 340, 460, 307, 35440, 351, 663]
VAL_TAIL = [1169, 2256, 6615, 13, 2398, 357, 17474, 8]

import datasets
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

print(f'datasets {datasets.__version__}, transformers {transformers.__version__}, numpy {np.__version__}')
print(f'out: {OUT}', flush=True)
os.makedirs(OUT, exist_ok=True)
train_path = os.path.join(OUT, f'openwebtext_train_{MAX_TRAIN_TOKENS // 1_000_000}M.npy')
val_path = os.path.join(OUT, f'openwebtext_val_{VAL_TOKENS // 1_000_000}M.npy')
assert not os.path.exists(train_path), f'{train_path} exists; move it away first'

tok = AutoTokenizer.from_pretrained('gpt2')
ds = load_dataset('Skylion007/openwebtext', split='train', streaming=True, trust_remote_code=True)
target = MAX_TRAIN_TOKENS + VAL_TOKENS
buf = np.empty(target + 200_000_000, dtype=np.uint16)   # one chunk overshoots the target by at most ~60M tokens


def pieces(s, size=1_000_000):
    """s cut into pieces of about `size` characters, each cut where a non-space is followed by a space or newline."""
    out, i, n = [], 0, len(s)
    while i < n:
        j = min(i + size, n)
        while j < n and not (s[j] in ' \n' and not s[j - 1].isspace()):
            j += 1
        out.append(s[i:j])
        i = j
    return out


def encode_chunk(texts):
    """The tokens of '\\n\\n'.join(texts), as GPT-2's encode gives them for the whole string."""
    joined = '\n\n'.join(texts)
    ps = pieces(joined)
    assert sum(map(len, ps)) == len(joined)
    out = []
    for g in range(0, len(ps), 32):
        for ids in tok(ps[g:g + 32], add_special_tokens=False)['input_ids']:
            out.append(np.asarray(ids, dtype=np.uint16))
    return np.concatenate(out)


total, n_docs, chunk_texts = 0, 0, []
t0 = time.time()
for example in ds:
    chunk_texts.append(example['text'])
    n_docs += 1
    if n_docs == 1:
        print(f'  first document streamed ({time.time() - t0:.0f}s)', flush=True)
    if len(chunk_texts) >= CHUNK_SIZE:
        ids = encode_chunk(chunk_texts)
        buf[total:total + len(ids)] = ids
        total += len(ids)
        chunk_texts = []
        el = time.time() - t0
        print(f'  {n_docs:,} docs  {total:,} tokens  ({el:.0f}s, {total / el / 1e6:.2f}M tok/s, '
              f'ETA {(target - total) / max(total / el, 1) / 60:.0f} min)', flush=True)
        if total >= target:
            break
if chunk_texts:                                        # only if the stream ran out first
    ids = encode_chunk(chunk_texts)
    buf[total:total + len(ids)] = ids
    total += len(ids)
print(f'Total streamed: {total:,} tokens from {n_docs:,} documents ({time.time() - t0:.0f}s)', flush=True)
val_ids = buf[total - VAL_TOKENS:total]
train_ids = buf[:min(total - VAL_TOKENS, MAX_TRAIN_TOKENS)]

np.save(val_path, val_ids)
file_md5 = hashlib.md5(open(val_path, 'rb').read()).hexdigest()
tok_md5 = hashlib.md5(val_ids.tobytes()).hexdigest()
match = tok_md5 == VAL_TOKENS_MD5
print(f'validation: {len(val_ids):,} tokens, file md5 {file_md5} ({"=" if file_md5 == VAL_FILE_MD5 else "!="} original), '
      f'token md5 {tok_md5} ({"=" if match else "!="} original)')
print(f'  head {val_ids[:8].tolist()} (original {VAL_HEAD})\n  tail {val_ids[-8:].tolist()} (original {VAL_TAIL})')
print('VERDICT: SAME POOL -- the stream reproduced; the training file below is the original.' if match else
      'VERDICT: DIFFERENT POOL -- the stream did not reproduce. The training file is written anyway; do not use it '
      'as the ladder pool without deciding so explicitly (the original validation file can be uploaded instead).',
      flush=True)
np.save(train_path, train_ids)
print(f'training: {len(train_ids):,} tokens -> {train_path} ({os.path.getsize(train_path):,} bytes)')
print('DONE')
