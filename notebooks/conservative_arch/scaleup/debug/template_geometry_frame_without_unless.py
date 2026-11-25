#!/usr/bin/env python3
"""template_geometry_frame.py with the wording "unless" left out of fitting and scoring (2026-11-25, POST HOC).
    python3 template_geometry_frame_without_unless.py RUNDIR OUTDIR stem[,stem...]"""
import sys
from pathlib import Path
T_ = Path(__file__).with_name('template_geometry_frame.py').read_text(); a_ = "dev, te = sp == 'dev', sp == 'test'"; assert T_.count(a_) == 1 and T_.count('template_geometry_frame.json') == 1
T_ = T_.replace(a_, "dev, te = (sp == 'dev') & (wi != 'unless'), (sp == 'test') & (wi != 'unless')").replace('template_geometry_frame.json', 'template_geometry_frame_without_unless.json').replace('out = {}\nfor s in STEMS', "out = {}; WC['disj'].remove('unless')\nfor s in STEMS")
assert "remove('unless')" in T_; exec(T_)
