"""Phrase-level octave planning; standard-library-only for deterministic tests."""
import math
import statistics

def consensus_pitch(normal,high,independent,tolerance_cents=60):
    """Reject high-register octave inflation only when two trackers agree."""
    result=[]
    for n,h,p in zip(normal,high,independent):
        choose=float(h)
        if n>0 and h>0 and p>0 and math.isfinite(p):
            agree=abs(1200*math.log2(p/n))<=tolerance_cents
            inflation=1200*math.log2(h/n)
            octave_error=abs(inflation-1200)<=100 or abs(inflation-1901.955)<=100
            if agree and octave_error:choose=float(n)
        result.append(choose)
    return result

def clean_pitch_track(f0,frame_rms,noise_floor,quiet_floor,min_island_frames=25):
    """Remove very faint pitch estimates and short, quiet residual islands.

    Strong short notes and sustained quiet notes are retained. Audio is untouched.
    """
    track=[float(f) if e>=noise_floor else 0. for f,e in zip(f0,frame_rms)]
    start=None
    for i,v in enumerate(track+[0.]):
        if v>0 and start is None:start=i
        if v<=0 and start is not None:
            if i-start<min_island_frames and max(frame_rms[start:i])<quiet_floor:
                track[start:i]=[0.]*(i-start)
            start=None
    return track

def register_ramps(plan):
    """Only bridge register changes inside actual source unvoiced gaps."""
    ramps=[]
    for i,r in enumerate(plan[1:],1):
        if not r['join_gap'] or r['semitones']==plan[i-1]['semitones']:continue
        a,b=r['join_gap'];a+=.02;b-=.02
        if b>a:ramps.append(dict(start=a,end=b,from_semitones=plan[i-1]['semitones'],to_semitones=r['semitones']))
    return ramps

def ramp_shift(time,ramp):
    u=max(0,min(1,(time-ramp['start'])/(ramp['end']-ramp['start'])))
    return ramp['from_semitones']+(ramp['to_semitones']-ramp['from_semitones'])*(.5-.5*math.cos(math.pi*u))

DEFAULTS = dict(center_hz=420, ceiling_p90_hz=750, ceiling_p99_hz=950,
                floor_median_hz=200, floor_p10_hz=130,
                pause_seconds=.18, min_phrase_seconds=1.2,
                transition_cost=6, continuity_window_seconds=8,
                octaves=[-24,-12,0,12,24])

def percentile(values, q):
    values=sorted(values)
    if not values:return 0.
    i=(len(values)-1)*q;lo=int(i);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(i-lo)

def plan_register(f0, duration, policy=None, fps=100):
    p={**DEFAULTS,**(policy or {})};f0=list(map(float,f0))
    if duration<=0 or not f0:raise ValueError('Empty pitch track')
    # Gaps, not timestamps or song names, define candidate joins.
    gaps=[];start=None
    for i,v in enumerate(f0+[1.]):
        if v<=0 and start is None:start=i
        if v>0 and start is not None:
            if start>0 and i<len(f0) and (i-start)/fps>=p['pause_seconds']:gaps.append((start/fps,i/fps))
            start=None
    cuts=[0.];joins={}
    for a,b in gaps:
        t=(a+b)/2
        if t-cuts[-1]>=p['min_phrase_seconds'] and duration-t>=p['min_phrase_seconds']:
            cuts.append(t);joins[t]=(a,b)
    cuts.append(duration);phrases=[]
    for a,b in zip(cuts,cuts[1:]):
        v=[x for x in f0[round(a*fps):round(b*fps)] if x>0]
        med=statistics.median(v) if v else 0.;q10=percentile(v,.1);q90=percentile(v,.9);q99=percentile(v,.99)
        candidates=[]
        for shift in p['octaves']:
            factor=2**(shift/12);m=med*factor;low=q10*factor;high=q90*factor;top=q99*factor
            safe=not v or (m>=p['floor_median_hz'] and low>=p['floor_p10_hz'] and high<=p['ceiling_p90_hz'] and top<=p['ceiling_p99_hz'])
            cost=abs(12*math.log2(max(m,1)/p['center_hz'])) if v else abs(shift)*.01
            candidates.append(dict(semitones=shift,safe=safe,cost=cost,target_median_hz=m,target_p90_hz=high,target_p99_hz=top))
        phrases.append(dict(start=a,end=b,source_median_hz=med,source_p90_hz=q90,source_voiced_frames=len(v),candidates=candidates,join_gap=joins.get(a)))
    # A continuity penalty discourages isolated octave toggles between adjacent phrases.
    states=p['octaves'];costs=[];backs=[]
    for i,phrase in enumerate(phrases):
        # Register fit is integrated over audible phrase length. A two-second
        # dip should not outweigh a long verse simply because both count once.
        duration_weight=max(.5,min(4,phrase['source_voiced_frames']/fps/3))
        local=[c['cost']*duration_weight if c['safe'] else math.inf for c in phrase['candidates']]
        if all(math.isinf(x) for x in local):
            raise ValueError(f'No safe octave fits phrase {phrase["start"]:.2f}–{phrase["end"]:.2f}s. Needs arrangement review.')
        if i==0:costs.append(local);backs.append([0]*len(states));continue
        next_cost=[];back=[]
        gap=phrase['join_gap']
        gap_seconds=gap[1]-gap[0] if gap else 0
        # Short breaths inside a musical line are expensive switching points;
        # longer section breaks permit more freedom. Cost follows location.
        location_weight=1+max(0,1-min(gap_seconds,1.5)/1.5)
        for j,state in enumerate(states):
            choices=[costs[-1][k]+p['transition_cost']*location_weight*abs(state-prev)/12 for k,prev in enumerate(states)]
            k=min(range(len(states)),key=lambda k:choices[k]);next_cost.append(local[j]+choices[k]);back.append(k)
        costs.append(next_cost);backs.append(back)
    j=min(range(len(states)),key=lambda j:costs[-1][j])
    for i in range(len(phrases)-1,-1,-1):
        phrases[i]['semitones']=states[j]
        if i:j=backs[i][j]
    # Eliminate brief excursions and returns when the surrounding register is
    # safe throughout that entire run. Never override a mandatory range limit.
    changed=True
    while changed:
        changed=False;start=0
        runs=[]
        for i in range(1,len(phrases)+1):
            if i==len(phrases) or phrases[i]['semitones']!=phrases[start]['semitones']:
                runs.append((start,i));start=i
        for j,(a,b) in enumerate(runs[1:-1],1):
            left=phrases[runs[j-1][0]]['semitones'];right=phrases[runs[j+1][0]]['semitones']
            seconds=phrases[b-1]['end']-phrases[a]['start']
            safe=all(any(c['safe'] and c['semitones']==left for c in q['candidates']) for q in phrases[a:b])
            if left==right and seconds<p['continuity_window_seconds'] and safe:
                for q in phrases[a:b]:
                    q['suppressed_short_register_excursion']=q['semitones'];q['semitones']=left
                changed=True;break
    return phrases

def retry_register_allowed(check):
    """Pitch-only errors with intact voicing should be polished in place."""
    return not (check.get('voiced_retention',0)>=.8 and check.get('voicing_agreement',0)>=.8
        and check.get('median_error_cents') is not None and check['median_error_cents']<=100)

def phrase_metrics(source_f0, output_f0, start, end, semitones, fps=100):
    # Exclude 50ms around phrase boundaries from the sustained-note check.
    i=round(start*fps)+5;j=min(round(end*fps)-5,len(source_f0),len(output_f0))
    if j<=i:return dict(passed=True,skipped='Phrase too short')
    pairs=list(zip(source_f0[i:j],output_f0[i:j]));voiced=[(float(r),float(y)) for r,y in pairs if r>0 and y>0]
    reference=sum(int(r>0) for r,y in pairs);generated=sum(int(r>0 and y>0) for r,y in pairs)
    if reference<20:return dict(passed=True,skipped='Insufficient voiced reference frames')
    errors=[abs(1200*math.log2(y/r)-100*semitones) for r,y in voiced]
    med=percentile(errors,.5) if errors else math.inf;p90=percentile(errors,.9) if errors else math.inf
    retained=generated/reference;agreement=sum(int((r>0)==(y>0)) for r,y in pairs)/len(pairs)
    return dict(passed=med<=50 and p90<=150 and retained>=.8 and agreement>=.8,
                median_error_cents=med if math.isfinite(med) else None,
                p90_error_cents=p90 if math.isfinite(p90) else None,
                voiced_reference_frames=reference,voiced_retention=retained,voicing_agreement=agreement)
