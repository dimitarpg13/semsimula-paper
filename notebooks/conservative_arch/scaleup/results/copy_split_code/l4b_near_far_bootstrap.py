import sys, numpy as np
from pathlib import Path
DBG=Path('/Users/dimitargueorguiev/git/ml/semsimula-paper/notebooks/conservative_arch/scaleup/debug'); sys.path.insert(0,str(DBG))
CSD=Path('/Users/dimitargueorguiev/git/ml/semsimula-paper/notebooks/conservative_arch/scaleup/results/copy_split_code')
src=(DBG/'copy_split.py').read_text(); src=src[:src.index("\nif MODE == 'nll':")]
sys.argv=['copy_split.py',str(CSD),'report']
G={'__name__':'cs','__file__':str(DBG/'copy_split.py')}; exec(compile(src,'copy_split.py','exec'),G)
x,y=G['windows'](); kind,cls,dist=G['categories'](x,y)
nl={k:np.load(CSD/f'nll_{k}.npy') for k in ('so','l4','gpt2','fock')}
W=nl['so'].shape[0]; pos=np.broadcast_to(np.arange(nl['so'].shape[1]),nl['so'].shape)
near=(kind>0)&(dist<=16); far=(kind>0)&(dist>64); mid=(kind>0)&(dist>16)&(dist<=64)
def per_window(m):
    # per-window sums for SO-L4 and SO-GPT2
    a=np.array([( (nl['so'][w]-nl['l4'][w])[m[w]].sum(), (nl['so'][w]-nl['gpt2'][w])[m[w]].sum(), (nl['so'][w]-nl['fock'][w])[m[w]].sum()) for w in range(W)]); return a
def closed(a,idx,j=0): s=a[idx].sum(0); return s[j]/s[1]
rng=np.random.default_rng(0)
for lab,extra in (('all positions',None),('positions 256-511',pos>=256)):
    mn=near if extra is None else near&extra; mf=far if extra is None else far&extra; mm=mid if extra is None else mid&extra
    an,af,am=per_window(mn),per_window(mf),per_window(mm)
    full=np.arange(W)
    for name,j in (('L4',0),('FOCK',2)):
        cn,cf,cm=closed(an,full,j),closed(af,full,j),closed(am,full,j)
        bs=[]
        for _ in range(4000):
            i=rng.integers(0,W,W); bs.append(closed(af,i,j)-closed(an,i,j))
        bs=np.array(bs)
        print(f'{lab:18s} {name}: closed 1-16 {100*cn:.1f}%  17-64 {100*cm:.1f}%  65-512 {100*cf:.1f}%  diff {100*(cf-cn):+.1f} pts  95% [{100*np.percentile(bs,2.5):+.1f}, {100*np.percentile(bs,97.5):+.1f}]  P(>0) {np.mean(bs>0):.3f}')
# identifier near/far
for nm,m in (('ident 1-16',(cls==1)&near),('ident 65-512',(cls==1)&far),('first occ',kind==0),('induction',kind==2),('indent',cls==0)):
    a=per_window(m); s=a.sum(0); print(f'{nm:14s} closed by L4 {100*s[0]/s[1]:.1f}%  (FOCK {100*s[2]/s[1]:.1f}%)  L4 gain/token {s[0]/m.sum():+.3f}')
s=per_window(np.ones_like(near)).sum(0); print('all closed L4 %.1f%% FOCK %.1f%%'%(100*s[0]/s[1],100*s[2]/s[1]))
print('L4 better than SO in %.1f%% of windows'%(100*np.mean(nl['l4'].mean(1)<nl['so'].mean(1))))
