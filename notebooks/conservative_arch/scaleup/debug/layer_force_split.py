import sys, numpy as np, torch
from pathlib import Path
DBG=Path('/Users/dimitargueorguiev/git/ml/semsimula-paper/notebooks/conservative_arch/scaleup/debug'); sys.path.insert(0,str(DBG))
kind,folder=sys.argv[1],sys.argv[2]; sys.argv=[sys.argv[0], sys.argv[3]]
import verify_pm_switch as P
torch.set_num_threads(8)
repl=P.CONFIGS['F3.1'][1]+P.PM_ON+(("LOWRANK_DAMPED_FLOW  = False","LOWRANK_DAMPED_FLOW  = True"),("POISSON_WELLS_EXACT  = False","POISSON_WELLS_EXACT  = True"),("POISSON_MODES        = 64","POISSON_MODES        = 16"))
if kind=='fob': repl+=(("FIRST_ORDER_OVERDAMPED = False","FIRST_ORDER_OVERDAMPED = True"),)
if 'nophi' in folder.split('_'): repl=tuple(r for r in repl if 'VPHI_GRAD_PATH' not in r[0])+(("PAIR_POTENTIAL  = 'sparse_topk'","PAIR_POTENTIAL  = 'none'"),)
m,g,_=P.build(folder,repl); m.eval(); m.cfg.use_layer_checkpoint=False
PHI=getattr(m,'V_phi',None) is not None; off=[False]
if PHI:
    pf,pp=m._pair_force_analytic,m._pair_potential
    m._pair_force_analytic=lambda h,*a,**k: torch.zeros_like(h) if off[0] else pf(h,*a,**k)
    m._pair_potential=lambda h,*a,**k: torch.zeros((),device=h.device) if off[0] else pp(h,*a,**k)
of=m._layer_forces; R={}; hin={}
def lf(h,xis,l,*,split=False,vtheta_comps=None):
    o=of(h,xis,l,split=split,vtheta_comps=vtheta_comps)
    if split and not off[0]:
        off[0]=True
        hh=h.detach().requires_grad_(True)
        with torch.enable_grad(): t,w=of(hh,xis,l,split=True,vtheta_comps=vtheta_comps)
        off[0]=False
        ft,fw=t.detach(),w.detach(); fp=((o[0]+o[1]).detach()-ft-fw)
        n=lambda x:x.norm(dim=-1); c=lambda a,b: torch.nn.functional.cosine_similarity(a,b,dim=-1)
        r=dict(theta=n(ft),wells=n(fw),cos_tw=c(ft,fw))
        h0=hin.get(l)
        if PHI: r.update(phi=n(fp),cos_pt=c(fp,ft),cos_pw=c(fp,fw),cos_prest=c(fp,ft+fw))
        for k,v in r.items(): R.setdefault((l,k),[]).append(v.reshape(-1).numpy())
        # the same theta force at the layer INPUT (where the conservativity test sits)
        if h0 is not None:
            hi=h0.detach().requires_grad_(True); off[0]=True
            with torch.enable_grad(): ti,wi=of(hi,xis,l,split=True,vtheta_comps=vtheta_comps)
            off[0]=False
            R.setdefault((l,'theta_at_input'),[]).append(n(ti.detach()).reshape(-1).numpy()); R.setdefault((l,'wells_at_input'),[]).append(n(wi.detach()).reshape(-1).numpy())
    return o
m._layer_forces=lf
oc=m._layer_step_langevin_core
def core(h,*a,**k):
    hin[a[4]]=h; return oc(h,*a,**k)
m._layer_step_langevin_core=core
rng=np.random.default_rng(20261009)
for _ in range(4):
    x,y=g['get_batch'](g['val_ids'],4,512,rng)
    with torch.enable_grad(): m(torch.from_numpy(x))
print(folder.split('norc_')[1].split('_L2probe')[0])
for l in (0,1): print('  layer',l,' '.join(f'{k}={np.median(np.concatenate(v)):.4g}' for (ll,k),v in R.items() if ll==l))
