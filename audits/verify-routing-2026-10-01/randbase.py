import sys, json, torch, numpy as np
S="./scratch"
sys.path[:0]=[S+"/zoology"]
from zoology.data.multiquery_ar import MQARConfig
from zoology.data.utils import DataSegment
run="./checkpoints/m4-L2048-s2-k4-baseline-seed231__20261001T044216Z"
man=json.load(open(run+"/manifest.json")); hs=man["data_seeds"]["heldout_test"]
torch.manual_seed(hs)
seg=DataSegment.from_config(MQARConfig(num_examples=2000,vocab_size=8192,input_seq_len=2048,num_kv_pairs=128),cache_dir=None,seed=hs)
X,Y=seg.inputs,seg.labels
qb,qt=(Y!=-100).nonzero(as_tuple=True)
keys=X[:,:256:2]; kpos=(keys[qb]==X[qb,qt].unsqueeze(-1)).float().argmax(-1)*2
kblk=(kpos//16).numpy(); ks=kblk//11; q=qt.numpy()
n=q//176                      # live supers: (s+1)*176 <= q
STR=[(256,352),(352,528),(528,1024),(1024,2048)]
print("heldout seeds same across runs:", len({json.load(open(p))["data_seeds"]["heldout_test"] for p in __import__('glob').glob('./checkpoints/*/manifest.json')}))
for lo,hi in STR:
    m=(q>=lo)&(q<hi)
    nn_=n[m]
    p1=np.where(nn_<=2,1.0,2.0/np.maximum(nn_,1))
    p4=1-(1-p1)**4
    # block-level random: per head, pick 2 random live supers (forced if <=2), then 4 random live candidate blocks among their members
    # simulate
    rng=np.random.default_rng(0)
    qs=q[m]; kb=kblk[m]; ksup=ks[m]
    sims1=[];sims4=[]
    idx=rng.choice(len(qs),size=min(len(qs),20000),replace=False)
    hit1=0;hit4=0
    for j in idx:
        qq=qs[j]; nl=qq//176
        live_sup=list(range(nl))
        # supers chosen: if nl>=2 random 2 of live; else live + super 1 (tie-break observed parent=1.0)
        heads=[]
        for h in range(4):
            if nl>=2: sel=rng.choice(live_sup,2,replace=False)
            else: sel=[0,1]
            cand=[b for s in sel for b in range(s*11,s*11+11) if (b+1)*16<=qq]
            pick=rng.choice(cand,min(4,len(cand)),replace=False)
            heads.append(kb[j] in pick)
        hit1+=heads[0]; hit4+=any(heads)
    print(f"[{lo},{hi}) frac={m.mean():.4f} n_live range {nn_.min()}-{nn_.max()} | parent random: 1-head {p1.mean():.3f}, 4 indep heads {p4.mean():.3f} | block random: 1-head {hit1/len(idx):.3f}, 4 indep heads {hit4/len(idx):.3f} | key in super1 {np.mean(ksup==1):.3f}")
    # n_live histogram
    u,c=np.unique(nn_,return_counts=True); print("    n_live dist:", dict(zip(u.tolist(), np.round(c/c.sum(),3).tolist())))
