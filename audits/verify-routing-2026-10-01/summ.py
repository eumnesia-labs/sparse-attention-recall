import json,glob
for f in sorted(glob.glob('verify3/probe_*.json')):
    d=json.load(open(f)); 
    print("=="*40); print(d['run'], 'acc',round(d['acc'],4))
    for li in (1,):
        print(f" L{li} supmatch {d[f'L{li}_supmatch']} recomp {d[f'L{li}_hit_recomp_vs_stored']} parent_any {d[f'L{li}_parent_any']:.3f} per-head {d[f'L{li}_parent_per_head']} block_any {d[f'L{li}_block_any']:.3f} per-head {d[f'L{li}_block_per_head']}")
        print("  hit@:", {k.split('hit_')[1]:v for k,v in d.items() if k.startswith(f'L{li}_hit_s')})
        for s in d[f'L{li}_strata']: print("  ",s)
    print(" recency", d["L1_late_pick_recency(0=most recent live super)"])
    print(" keysuper rank", d["L1_late_keysuper_rank_head0_dist"])
    for k,v in d['L1_fallback'].items(): print("  fb",k,v)
