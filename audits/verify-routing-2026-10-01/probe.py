import sys, os, json, inspect, math, collections, torch, numpy as np
S="./scratch"
sys.path[:0]=[S+"/zoology", S+"/verify3/src"]
import zoology.model as zm
from zoology.config import ModelConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.data.utils import DataSegment
from zoology.model import LanguageModel
import hier_nsa
from hier_nsa import HierSparseAttention, NEG
for _, c in inspect.getmembers(zm, inspect.isclass):
    d = getattr(c.__init__, "__defaults__", None)
    if d and "cuda" in d: c.__init__.__defaults__ = tuple("cpu" if x == "cuda" else x for x in d)
torch.set_grad_enabled(False)
run=sys.argv[1]; N=int(sys.argv[2]); assert N<=64
man=json.load(open(run+"/manifest.json")); P=man["protocol"]
s1=man.get("s1") or P["s1"]; cb=P["compress_block_size"]; g=P["superblock_group"]; h=4
mixer=ModuleConfig(name="hier_nsa.HierSparseAttention", kwargs={"num_heads":4,"sliding_window_size":P["sliding_window_size"],"compress_block_size":cb,"selection_block_size":cb,"num_selected_blocks":P["k_sel"],"superblock_group":g,"num_selected_superblocks":s1,"causal":True,**(man.get("arm_kwargs") or {})})
model=LanguageModel(ModelConfig(vocab_size=8192,max_position_embeddings=2048,d_model=128,n_layers=2,sequence_mixer=mixer))
model.load_state_dict(torch.load(run+"/best.pt",map_location="cpu"),strict=True); model.eval()
mods=[m for m in model.modules() if isinstance(m,HierSparseAttention)]
cap={}
for li,m in enumerate(mods):
    m.keep_routing=True
    m.qkv.register_forward_hook(lambda mod,i,o,li=li: cap.__setitem__(li,o))
kill={"layer":None,"br":None}
def ghook(mod,i,o,li):
    if kill["layer"]!=li: return o
    oo=o.clone().view(o.shape[0],o.shape[1],4,3); oo[...,kill["br"]]=-1e4; return oo.view_as(o)
for li,m in enumerate(mods): m.gate.register_forward_hook(lambda mod,i,o,li=li: ghook(mod,i,o,li))
hs=man["data_seeds"]["heldout_test"]; torch.manual_seed(hs)
seg=DataSegment.from_config(MQARConfig(num_examples=2000,vocab_size=8192,input_seq_len=2048,num_kv_pairs=128),cache_dir=None,seed=hs)
X,Y=seg.inputs[:N],seg.labels[:N]
SK=[(2,4),(2,8),(2,16),(4,4),(4,8),(6,4),(12,4),(12,8),(12,16)]
acc=collections.defaultdict(list)
for i in range(0,N,8):
    x,y=X[i:i+8],Y[i:i+8]; B,L=x.shape
    kill["layer"]=None; lg=model(x)
    qb,qt=(y!=-100).nonzero(as_tuple=True)
    keys=x[:,:256:2]; kpos=(keys[qb]==x[qb,qt].unsqueeze(-1)).float().argmax(-1)*2; kblk=kpos//cb; ksup=kblk//g
    corr=(lg[qb,qt].argmax(-1)==y[qb,qt])
    acc["qt"].append(qt); acc["corr"].append(corr); acc["kblk"].append(kblk)
    for li,m in enumerate(mods):
        o=cap[li]; q,k,v=o.reshape(B,L,3,h,-1).permute(2,0,3,1,4); hd=q.shape[-1]; sc=1/math.sqrt(hd)
        bk,nblk=hier_nsa._block_pool(k,cb); sk,nsup=hier_nsa._block_pool(bk,g,m.superblock_pool)
        Q=q[qb,:,qt]                                        # (NQ,h,hd)
        ss=torch.einsum("nhd,nhsd->nhs",Q,sk[qb])*sc
        sup_end=(torch.arange(nsup)+1)*g*cb
        ss=ss.masked_fill(sup_end[None,None,:]>qt[:,None,None],NEG)
        stored_sup=m.last_sup_idx[qb,:,qt]; stored_fin=m.last_fin_blk[qb,:,qt]
        mine=ss.topk(min(s1,nsup),dim=-1).indices
        acc[f"L{li}_supmatch"].append((mine.sort(-1).values==stored_sup.sort(-1).values).all(-1).all(-1))
        bs=torch.einsum("nhd,nhbd->nhb",Q,bk[qb])*sc       # (NQ,h,nblk)
        blive=((torch.arange(nblk)+1)*cb)[None,None,:]<=qt[:,None,None]
        kscore=bs.gather(-1,kblk[:,None,None].expand(-1,h,1)).squeeze(-1)   # (NQ,h)
        bsup=torch.arange(nblk)//g
        # per head parent hit, block hit (stored)
        ph=(stored_sup==ksup[:,None,None]).any(-1); bh=(stored_fin==kblk[:,None,None]).any(-1)
        acc[f"L{li}_ph"].append(ph); acc[f"L{li}_bh"].append(bh)
        dmin=torch.where(stored_fin>=kblk[:,None,None], stored_fin-kblk[:,None,None], torch.full_like(stored_fin,10**6)).amin(-1).amin(-1)
        acc[f"L{li}_dmin"].append(dmin)
        # selected-super recency for live picks: rel = (n_live-1)-sup
        nl=(qt//(g*cb))
        acc[f"L{li}_rel"].append((nl[:,None,None]-1-stored_sup))
        acc[f"L{li}_suprank"].append((ss> ss.gather(-1,ksup[:,None,None].expand(-1,h,1))).sum(-1))  # rank of key super per head
        for (s,K) in SK:
            sel=ss.topk(min(s,nsup),dim=-1).indices      # (NQ,h,s)
            cand=(bsup[None,None,:,None]==sel[:,:,None,:]).any(-1) & blive   # (NQ,h,nblk)
            keyc=(sel==ksup[:,None,None]).any(-1)
            rank=((bs>kscore[...,None]) & cand).sum(-1)
            hitK=keyc & (rank<K)
            acc[f"L{li}_hit_s{s}_K{K}"].append(hitK)
    # ablations at layer 1
    for br,name in ((0,"nocoarse"),(1,"nofine"),(2,"nowin")):
        kill["layer"]=1; kill["br"]=br; lg2=model(x); kill["layer"]=None
        acc[f"corr_{name}"].append(lg2[qb,qt].argmax(-1)==y[qb,qt])
R={k:torch.cat(v) for k,v in acc.items()}
qt=R["qt"]; corr=R["corr"].float(); nq=len(qt)
out={"run":os.path.basename(run),"nq":nq,"acc":corr.mean().item()}
STR=[(256,352),(352,528),(528,1024),(1024,2048)]
for li in range(2):
    ph=R[f"L{li}_ph"]; bh=R[f"L{li}_bh"]
    out[f"L{li}_supmatch"]=R[f"L{li}_supmatch"].float().mean().item()
    out[f"L{li}_parent_any"]=ph.any(-1).float().mean().item()
    out[f"L{li}_parent_per_head"]=[round(v,3) for v in ph.float().mean(0).tolist()]
    out[f"L{li}_block_any"]=bh.any(-1).float().mean().item()
    out[f"L{li}_block_per_head"]=[round(v,3) for v in bh.float().mean(0).tolist()]
    # validate recomputed (s1,4) vs stored
    out[f"L{li}_hit_recomp_vs_stored"]=(R[f"L{li}_hit_s{s1}_K4"].any(-1)==bh.any(-1)).float().mean().item()
    for (s,K) in SK: out[f"L{li}_hit_s{s}_K{K}"]=round(R[f"L{li}_hit_s{s}_K{K}"].any(-1).float().mean().item(),4)
    st=[]
    for lo,hi in STR:
        mk=(qt>=lo)&(qt<hi)
        st.append({"str":f"[{lo},{hi})","parent_any":round(ph[mk].any(-1).float().mean().item(),3),
                   "parent_per_head":[round(v,3) for v in ph[mk].float().mean(0).tolist()],
                   "block_any":round(bh[mk].any(-1).float().mean().item(),3),
                   **{f"s{s}K{K}":round(R[f"L{li}_hit_s{s}_K{K}"][mk].any(-1).float().mean().item(),3) for (s,K) in SK}})
    out[f"L{li}_strata"]=st
# recency of layer-1 parent picks for q>=528
late=qt>=528
rel=R["L1_rel"][late]
c=collections.Counter(rel.flatten().tolist()); tot=sum(c.values())
out["L1_late_pick_recency(0=most recent live super)"]={k:round(v/tot,3) for k,v in sorted(c.items())[:12]}
# key-super rank among supers at layer 1 (late)
sr=R["L1_suprank"][late]
out["L1_late_keysuper_rank_head0_dist"]={k:round(v/len(sr),3) for k,v in sorted(collections.Counter(sr[:,0].tolist()).items())}
# fallback analysis at layer 1
bh1=R["L1_bh"].any(-1); ph1=R["L1_ph"].any(-1); dmin=R["L1_dmin"]
def a(mask, key="corr"): 
    return (round(R[key][mask].float().mean().item(),3), int(mask.sum())) if mask.any() else None
fb={}
for nm,mk in (("hit",bh1),("miss",~bh1),("miss&parent_hit",~bh1&ph1),("miss&parent_miss",~bh1&~ph1),
              ("miss&off1-2",~bh1&(dmin>=1)&(dmin<=2)),("miss&off>=3/none",~bh1&(dmin>=3))):
    fb[nm]={"full":a(mk),"no_coarse_L1":a(mk,"corr_nocoarse"),"no_fine_L1":a(mk,"corr_nofine"),"no_win_L1":a(mk,"corr_nowin")}
out["L1_fallback"]=fb
print(json.dumps(out,indent=1))
