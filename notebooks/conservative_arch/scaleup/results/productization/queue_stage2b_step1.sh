#!/bin/bash
# Stage 2b, step 1 (companion_notes/productization/Stage2b_Decode_Step_Levers.md, §3.3), 2026-11-23. One job at a time.
cd "$(dirname "$0")/../.."
V=$HOME/Downloads/semsimula_fock_aniso_gaussian_fockreg_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1/data/openwebtext_val_2M.npy
F=semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn
O=results/productization
L=$O/queue_stage2b_step1.log
COMMON="lengths=512,2048,8192 decode=1 decode_tokens=32 chunk=512 skipreg=1 gram=eigh64 loss=$V"
run () { name=$1; shift; python3 debug/cpu_throughput_benchmark.py $O ladder $F $COMMON "$@" name=$name > $O/decode_ab_$name.stdout 2>&1; echo "done $name $?" >> $L; }
: > $L
run s1_null_pruned      prune=1:5 ab=base,base
run s1_masked_pruned    prune=1:5 ab=base,masked ab_prefill=3
run s1_null_unpruned              ab=base,base
run s1_masked_unpruned            ab=base,masked ab_prefill=3
run s1_auto_unpruned              ab=masked,masked+auto
echo "done all" >> $L
