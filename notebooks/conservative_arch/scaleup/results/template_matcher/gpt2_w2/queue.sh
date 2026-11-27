cd /Users/dimitargueorguiev/git/ml/semsimula-paper/notebooks/conservative_arch/scaleup
PY=/Users/dimitargueorguiev/.pyenv/versions/3.12.11/bin/python3; G=semsimula_matched_gpt2_baseline_owt_wsd_lr0p0012; SP=$SCRATCH/gp; R=results/template_matcher; Q=$SP/q.log; : > $Q
run() { n=$1; shift; echo "$(date +%T) start $n" >> $Q; "$@" > $SP/$n.txt 2> $SP/$n.err; echo "$(date +%T) $n rc=$?" >> $Q; }
run ext_vec   $PY debug/template_vectors_extract.py $SP/vec $G name=w2
run depth     $PY debug/template_depth_analysis.py $SP/vec $R w2
run lexical   $PY debug/template_lexical_baseline.py $SP/vec $R w2
run ext_t2    $PY debug/template_analogy_test.py $SP/t2 $G name=w2 vec=1
run ext_bl    $PY debug/template_analogy_test.py $SP/bl $G name=w2 vec=1 bank=blogic
run ext_bc    $PY debug/template_analogy_test.py $SP/bc $G name=w2 vec=1 bank=blogic_control
run gates_t2  $PY debug/template_analogy_analysis.py $SP/vec $SP/t2 $R w2 gates
run gates_bl  $PY debug/template_analogy_analysis.py $SP/vec $SP/bl $R w2 gates items=template_blogic_items tag=blogic
run gates_bc  $PY debug/template_analogy_analysis.py $SP/vec $SP/bc $R w2 gates items=template_blogic_control_items tag=blogic_control
run taskA     $PY debug/template_summary_state.py $SP/vec $R w2 analogy=$SP/t2
run sum_bl    $PY debug/template_summary_state.py $SP/vec $R w2 analogy=$SP/bl items=template_blogic_items tag=blogic
run sum_bc    $PY debug/template_summary_state.py $SP/vec $R w2 analogy=$SP/bc items=template_blogic_control_items tag=blogic_control
run score_t2  $PY debug/template_analogy_analysis.py $SP/vec $SP/t2 $R w2
run score_bl  $PY debug/template_analogy_analysis.py $SP/vec $SP/bl $R w2 items=template_blogic_items tag=blogic
run score_bc  $PY debug/template_analogy_analysis.py $SP/vec $SP/bc $R w2 items=template_blogic_control_items tag=blogic_control
echo "$(date +%T) all done" >> $Q
