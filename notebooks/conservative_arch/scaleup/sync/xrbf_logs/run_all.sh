#!/bin/zsh
cd "$(dirname "$0")/.."
for s in sy_measure sy_routes sy_xi_pred sy_coref_probe sy_subspace sy_coref_subspace_check sy5_depth sy_exch_reference; do
  echo "=== $s start $(date +%H:%M:%S)" >> xrbf_logs/progress.txt
  python3 $s.py XRBF > xrbf_logs/$s.log 2>&1
  echo "=== $s exit $? $(date +%H:%M:%S)" >> xrbf_logs/progress.txt
done
echo ALLDONE >> xrbf_logs/progress.txt
