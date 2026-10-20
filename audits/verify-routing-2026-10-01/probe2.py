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

res={"pick_in_01":[], "both_in_01":[], "keysup":[], "ph":[]}
for i in range(0,N,8):
    x,y=X[i:i+8],Y[i:i+8]
    kill["layer"]=None; lg=model(x)
    qb,qt=(y!=-100).nonzero(as_tuple=True)
    keys=x[:,:256:2]; kpos=(keys[qb]==x[qb,qt].unsqueeze(-1)).float().argmax(-1)*2; ksup=(kpos//cb)//g
    late=qt>=528
    sup=mods[1].last_sup_idx[qb,:,qt][late]          # (n,h,2)
    res["pick_in_01"].append((sup<=1).float().mean((1,2)))
    res["both_in_01"].append((sup<=1).all(-1).float().mean(-1))
    res["ph"].append((sup==ksup[late][:,None,None]).any(-1).any(-1).float())
import torch
R={k:torch.cat(v) for k,v in res.items() if v}
print(os.path.basename(run), "L1 q>=528: frac of picks in supers{0,1}=%.3f  frac heads picking exactly {0,1}=%.3f  parent_any=%.3f" % (R["pick_in_01"].mean(), R["both_in_01"].mean(), R["ph"].mean()))
