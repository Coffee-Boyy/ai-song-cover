"""Conservative waveform cleanup and source-aware whole-track diagnostics."""
import numpy as np
from scipy.ndimage import median_filter,uniform_filter1d,gaussian_filter1d
from scipy.signal import butter,sosfiltfilt,resample_poly
from math import gcd

def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]))

def match_source(source,source_sr,sr,n):
    x=np.asarray(source,dtype=float)
    if x.ndim>1:x=x.mean(1)
    if source_sr!=sr:
        g=gcd(source_sr,sr);x=resample_poly(x,sr//g,source_sr//g)
    return np.pad(x,(0,max(0,n-len(x))))[:n]

def impulse_residual(x):
    r=np.zeros(len(x));r[1:-1]=x[1:-1]-(x[:-2]+x[2:])/2
    return r

def cleanup_audio(audio,sr,source=None,source_sr=None):
    original=np.asarray(audio,dtype=float);y=original.copy()
    if y.ndim!=1 or not np.isfinite(y).all():raise ValueError('Expected finite mono vocal')
    if len(y)<sr//2:raise ValueError('Cleanup requires at least half a second')
    reference=match_source(source,source_sr,sr,len(y)) if source is not None else None
    r=impulse_residual(y);local=median_filter(abs(r),size=max(5,round(sr*.002)//2*2+1))
    candidates=runs(abs(r)>np.maximum(.025,18*local));clicks=[]
    candidate_starts=np.array([a for a,b in candidates])
    rr=impulse_residual(reference) if reference is not None else None
    for a,b in candidates:
        # Only microsecond-scale outliers: do not treat sustained harshness as clicks.
        if b-a>round(sr*.0005) or a<3 or b>len(y)-3:continue
        left=max(0,a-round(sr*.002));right=min(len(y),b+round(sr*.002))
        if rr is not None:
            span=max(1,round(sr*.02));u=max(0,a-span);v=min(len(y),b+span)
            ratio=np.sqrt(np.mean(y[u:v]**2))/max(np.sqrt(np.mean(reference[u:v]**2)),1e-8)
            if max(abs(rr[left:right]))*ratio>=max(abs(r[a:b]))*.35:continue
        # Repeated sharp vocal pulses are tonal material, not isolated clicks.
        nearby=np.searchsorted(candidate_starts,a+sr*.01,side='right')-np.searchsorted(candidate_starts,a-sr*.01)
        if nearby>2:continue
        a-=1;b+=1
        y[a:b]=np.interp(np.arange(a,b),[a-1,b],[y[a-1],y[b]])
        clicks.append(dict(start=a/sr,end=b/sr))
    # Subsonic cleanup only if the rendered vocal actually contains meaningful DC/rumble.
    filtered=sosfiltfilt(butter(2,25,btype='highpass',fs=sr,output='sos'),y)
    removed_ratio=float(np.mean((y-filtered)**2)/max(np.mean(y*y),1e-12))
    dc=float(np.mean(y));subsonic=abs(dc)>.0001 or removed_ratio>.005
    if subsonic:y=filtered
    # Level diagnostics use 10ms frames and source-relative context.
    hop=max(1,round(sr*.01));frames=len(y)//hop
    rms=np.sqrt(np.mean(y[:frames*hop].reshape(frames,hop)**2,axis=1)+1e-14)
    db=20*np.log10(rms);base=median_filter(db,size=31);deviation=db-base
    spikes=[];gain=np.ones(frames);dropouts=[]
    if reference is not None:
        ref_rms=np.sqrt(np.mean(reference[:frames*hop].reshape(frames,hop)**2,axis=1)+1e-14)
        ref_db=20*np.log10(ref_rms);ref_base=median_filter(ref_db,size=31);ref_dev=ref_db-ref_base
        audible=ref_db>max(-55,float(np.percentile(ref_db,95))-35)
        for a,b in runs((deviation>9)&(ref_dev<3)&audible):
            if 2<=b-a<=10:
                gain[a:b]=10**(-np.minimum(deviation[a:b]-3,6)/20)
                spikes.append(dict(start=a*.01,end=b*.01,max_attenuation_db=float(-20*np.log10(gain[a:b].min()))))
        # Report losses of voiced energy; do not invent missing samples or syllables.
        for a,b in runs((deviation<-18)&(ref_dev>-3)&audible):
            if b-a>=5:dropouts.append(dict(start=a*.01,end=b*.01))
    gain=gaussian_filter1d(gain,sigma=1.5)
    envelope=np.interp(np.arange(len(y))/sr,np.arange(frames)*.01,gain,left=gain[0],right=gain[-1]);y*=envelope
    if not np.isfinite(y).all() or len(y)!=len(original):raise ValueError('Cleanup changed duration or produced invalid samples')
    report=dict(declick_repairs=clicks,level_spike_repairs=spikes,subsonic_filter_applied=bool(subsonic),subsonic_removed_energy_ratio=removed_ratio,input_dc_offset=dc,dropouts_requiring_review=dropouts,output_peak=float(abs(y).max()),duration_seconds=len(y)/sr,processing='Isolated impulse interpolation; conditional 25Hz zero-phase subsonic removal; source-protected level spike attenuation capped at 6dB with smooth gain. No gate, broad denoise, time stretching or missing-vowel synthesis.')
    return y,report

def audit_pitch_islands(target,generated):
    n=min(len(target),len(generated));t=np.asarray(target[:n]);g=np.asarray(generated[:n]);valid=(t>0)&(g>0)
    errors=np.zeros(n);errors[valid]=1200*np.log2(g[valid]/t[valid])
    suspicious=[]
    for a,b in runs(valid&(abs(errors)>600)):
        if b-a<8:continue
        if a==0 or b==n or not valid[max(0,a-3):a].all() or not valid[b:min(n,b+3)].all():continue
        # If the intended melody itself changes register, do not classify it as a glitch.
        intended=1200*np.log2(t[a:b]/np.median(t[max(0,a-3):a]))
        if np.max(abs(intended))>250:continue
        suspicious.append(dict(start=a*.01,end=b*.01,median_error_cents=float(np.median(abs(errors[a:b])))))
    return suspicious

def true_peak(audio,factor=4):
    x=np.asarray(audio)
    if not np.isfinite(x).all():raise ValueError('Nonfinite export samples')
    return float(abs(resample_poly(x,factor,1,axis=0)).max())
