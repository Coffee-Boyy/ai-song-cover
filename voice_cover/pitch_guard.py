"""Repair a source melody only where two independent trackers support a correction."""
import numpy as np

def repair_source_guide(guide,stock,praat,strength,agreement_cents=60,
                        deviation_cents=180,min_frames=4,min_strength=.6):
    guide=np.asarray(guide,dtype=float).copy()
    n=min(len(guide),len(stock),len(praat),len(strength))
    stock=np.asarray(stock)[:n];praat=np.asarray(praat)[:n]
    strength=np.asarray(strength)[:n]
    valid=(stock>0)&(praat>0)&np.isfinite(praat)&(strength>=min_strength)&(guide[:n]>0)
    agreement=np.full(n,np.inf);deviation=np.zeros(n)
    agreement[valid]=abs(1200*np.log2(stock[valid]/praat[valid]))
    deviation[valid]=abs(1200*np.log2(guide[:n][valid]/stock[valid]))
    reliable=valid&(agreement<=agreement_cents)
    # Require sustained independent agreement, not sustained error. This also
    # repairs short chirps embedded inside an otherwise well-supported note.
    edges=np.flatnonzero(np.diff(np.r_[False,reliable,False]))
    supported=np.zeros(n,dtype=bool)
    for a,b in zip(edges[::2],edges[1::2]):
        if b-a>=min_frames:supported[a:b]=True
    candidate=supported&(deviation>=deviation_cents)
    edges=np.flatnonzero(np.diff(np.r_[False,candidate,False]))
    events=[]
    for a,b in zip(edges[::2],edges[1::2]):
        events.append(dict(start_seconds=float(a*.01),end_seconds=float(b*.01),
            frames=int(b-a),median_disagreement_cents=float(np.median(deviation[a:b])),
            independent_agreement_p90_cents=float(np.percentile(agreement[a:b],90))))
        guide[a:b]=stock[a:b]
    return guide,dict(enabled=True,corrected_frames=sum(e['frames'] for e in events),
        events=events,agreement_cents=agreement_cents,min_praat_strength=min_strength,
        minimum_independent_agreement_ms=min_frames*10,method='Stock RMVPE + independent Praat agreement replaces unsupported source-guide excursions before register planning; no chromatic snapping.')
