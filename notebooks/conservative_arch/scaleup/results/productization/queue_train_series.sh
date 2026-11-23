#!/bin/bash
# Does the series length of the damped-mode coefficients matter in TRAINING? CPU estimate (2026-11-23,
# Stage2b_Decode_Step_Levers.md §6). Stop rule agreed with the author: under about 2% of a training step, no GPU session.
cd "$(dirname "$0")/../.."
E2E=results/productization/queue_stage2b_e2e.log
until grep -q "done all" $E2E 2>/dev/null; do sleep 20; done          # one CPU job at a time
V=$HOME/Downloads/semsimula_fock_aniso_gaussian_fockreg_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1/data/openwebtext_val_2M.npy
F=semsimula_fock_cfc_baoab_owt_xi5long_topk16_dt32da16_mh4_aniso_dcvt5x8_vtjoint_cgqk_norc_nophi_xilive_sr2_L2probe_ob_untied_wsd_e5c_plgate_rep0.05_fockreg0.005_g0.1_baoab_cfc_lowrank_idt4_lr0p0012_noattn
O=results/productization
L=$O/queue_train_series.log
COMMON="lengths=512 train_batch=4 train_T=512 train_blocks=6 loss=$V"
run () { name=$1; shift; python3 debug/cpu_throughput_benchmark.py $O ladder $F $COMMON "$@" name=$name > $O/train_ab_$name.stdout 2>&1; echo "done $name $?" >> $L; }
: > $L
run ts_null   train_ab=40,40
run ts_auto   train_ab=40,auto
run ts_zero   train_ab=40,0
echo "done all" >> $L
