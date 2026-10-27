#!/usr/bin/env python3
"""Rebuild the code validation tokens (code_val_2M.npy) locally, for copy_split.py (2026-10-26).

    python3 copy_split_build_val.py OUT.npy

Runs the ladder notebook's own code-corpus functions (Cell 3, CORPUS = 'code': the tab rewrite of leading
indentation and the chunked tokeniser) on the pinned codeparrot-clean-valid revision, read with
hf_hub_download and gzip, because `datasets` does not import in the local Python (no _lzma). The first
2,000,000 tokens match the Drive cache the code runs trained against: GPT-2-CODE gives 5.04 on them, against
4.95 logged on Colab's random batches.
"""
import gzip, importlib.machinery, json, sys, time, types
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

NB = Path(__file__).resolve().parents[1] / 'colab_fock_cfc_baoab_lowrank_depth_ladder_openwebtext_d384.ipynb'
cell = next(''.join(c['source']) for c in json.load(open(NB))['cells']
            if ''.join(c['source']).startswith('# == Cell 3:'))
a = cell.index("elif CORPUS == 'code':"); b = cell.index('    from transformers import AutoTokenizer', a)
body = '\n'.join(l[4:] for l in cell[a:b].split('\n')[1:])     # the branch's own definitions, dedented


def load_dataset(name, split, streaming, revision):
    assert (name, split, streaming) == ('codeparrot/codeparrot-clean-valid', 'train', True)
    shard = hf_hub_download(name, 'file-000000000054.json.gz', repo_type='dataset', revision=revision)

    def gen():
        with gzip.open(shard, 'rt') as f:
            for line in f:
                yield json.loads(line)
    return gen()


m = types.ModuleType('datasets'); m.__spec__ = importlib.machinery.ModuleSpec('datasets', None)
m.load_dataset = load_dataset; sys.modules['datasets'] = m
from transformers import AutoTokenizer
g = {'np': np, 'time': time, 'tok': AutoTokenizer.from_pretrained('gpt2')}
exec(body, g)
val = g['_code_tokens'](g['CODE_VALID_REPO'], g['VAL_TOKENS'])
np.save(sys.argv[1], val)
print(f'{len(val):,} tokens ({val.dtype}) -> {sys.argv[1]}')
