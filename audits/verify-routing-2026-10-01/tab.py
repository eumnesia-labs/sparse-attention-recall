import json
d=json.load(open('routing-diagnostic-2026-10-01.json'))
for r in d:
    print("="*100)
    print(r['run'], r['arm'], 's1=',r['s1'], 'seed',r['seed'], 'nseq',r['n_sequences'],'Q',r['n_queries'],
          'acc',r['accuracy_reproduced'],'rec',r['recorded_primary'],'gate',r['validity_gate'],'win',r['value_in_window_frac'])
    for L in r['layers']:
        bh=L['block_hit_any_head']; ah=L['acc_given_block_hit']; am=L['acc_given_block_miss']
        recon = (bh*(ah or 0) + (1-bh)*(am or 0)) if ah is not None else None
        print(f" L{L['layer']} parent={L['parent_hit_any_head']:.4f} blk={bh:.4f} perhead={[round(x,3) for x in L['block_hit_per_head']]} acc|hit={ah} acc|miss={am} recon={recon} diff={None if recon is None else recon-r['accuracy_reproduced']:.2e}")
        offs=L['nearest_selected_offset']
        print("   offs:", {k:(round(v['frac'],4), None if v['acc'] is None else round(v['acc'],4)) for k,v in offs.items()}, 'sumfrac',round(sum(v['frac'] for v in offs.values()),6))
        # offset-recon
        oacc=sum(v['frac']*(v['acc'] or 0) for v in offs.values())
        print("   offs recon acc", round(oacc,6))
        fsum=0; asum=0; psum=0; bsum=0
        for s in L['by_query_position']:
            fsum+=s['frac_queries']; asum+=s['frac_queries']*s['acc']; psum+=s['frac_queries']*s['parent_hit_any_head']; bsum+=s['frac_queries']*s['block_hit_any_head']
            print(f"   [{s['q_from']},{s['q_to']}) f={s['frac_queries']:.4f} acc={s['acc']:.4f} par={s['parent_hit_any_head']:.4f} blk={s['block_hit_any_head']:.4f} acc|hit={s['acc_given_hit']} acc|miss={s['acc_given_miss']}")
        print(f"   strata sums: f={fsum:.6f} acc={asum:.6f} par={psum:.6f} blk={bsum:.6f}")
