"""Source-adaptive vocal production, using only the generated voice for effect sends."""
import json
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d,gaussian_filter1d
from scipy.signal import butter,sosfiltfilt

def analyze_effects(reference,foreground,sr):
    if reference.ndim==1:reference=reference[:,None]
    n=min(len(reference),len(foreground));reference=reference[:n];foreground=foreground[:n]
    mid=reference.mean(1);side=(reference[:,0]-reference[:,-1])*.5
    hop=max(1,round(.02*sr))
    def envelope(x):return np.sqrt(np.maximum(uniform_filter1d(x*x,size=hop)[::hop],0)+1e-12)
    dry= envelope(foreground);wet=envelope(mid);sides=envelope(side)
    active=dry>np.percentile(dry,95)*.15
    if not active.any():raise ValueError('No foreground activity for production analysis')
    width=float(np.sqrt(np.mean(sides[active]**2))/max(np.sqrt(np.mean(wet[active]**2)),1e-9))
    stops=np.where(active[:-1]&~active[1:])[0]+1;tails=[]
    for i in stops:
        if i+35>=len(dry) or np.any(active[i+3:i+30]):continue
        ids=np.arange(i+3,i+30);db=20*np.log10(wet[ids]+1e-9);t=ids*.02
        slope=float(np.polyfit(t,db,1)[0]);corr=float(np.corrcoef(t,db)[0,1])
        if slope<-5 and corr<-.7:tails.append(dict(time=float(i*.02),rt60_seconds=float(-60/slope),correlation=corr))
    rt60=float(np.clip(np.median([q['rt60_seconds'] for q in tails]),.3,4)) if tails else .65
    # Late energy relative to phrase energy controls the send rather than a
    # single fixed wet setting. Separation bleed makes this an approximation.
    ratios=[]
    for tail in tails:
        i=round(tail['time']/.02);lead=wet[max(0,i-20):i]
        late=wet[i+5:min(len(wet),i+25)]
        if len(lead) and len(late):ratios.append(float(np.sqrt(np.mean(late**2))/max(np.sqrt(np.mean(lead**2)),1e-9)))
    send=float(np.clip(np.median(ratios),.025,.22)) if ratios else .025
    crest_ref=float(np.percentile(abs(mid),99.9)/max(np.sqrt(np.mean(mid**2)),1e-9))
    # Rhythm is measured from the supplied recording; this is not an assertion
    # that the source used a particular delay plug-in or tap pattern.
    import librosa
    a=librosa.resample(mid,orig_sr=sr,target_sr=11025).astype(np.float32)
    onset=librosa.onset.onset_strength(y=a,sr=11025)
    tempo=float(np.asarray(librosa.beat.beat_track(onset_envelope=onset,sr=11025)[0]).ravel()[0])
    if not np.isfinite(tempo) or tempo<55 or tempo>200:tempo=0
    return dict(reference_rt60_estimate_seconds=rt60,tail_evidence=tails,
        stereo_side_mid_ratio=width,reverb_send=send,source_crest_factor=crest_ref,
        estimated_tempo_bpm=tempo,delay_seconds=30/tempo if tempo else 0,
        delay_send=min(.035,send*.15) if tempo else 0,
        limitations='Production approximation from separated source; effects settings and harmony parts cannot be recovered exactly. Tempo-linked delay and generated doubles are restrained approximations.')

def apply_effects(vocal,sr,reference_path,foreground_path):
    from pedalboard import Pedalboard,Compressor,HighpassFilter,LowpassFilter,Reverb
    reference,rsr=sf.read(reference_path,always_2d=True);foreground,fsr=sf.read(foreground_path,always_2d=True)
    from scipy.signal import resample_poly
    from math import gcd
    def resample(x,s):
        g=gcd(s,sr);return resample_poly(x,sr//g,s//g,axis=0) if s!=sr else x
    reference=resample(reference,rsr);foreground=resample(foreground,fsr).mean(1)
    settings=analyze_effects(reference,foreground,sr)
    voice=np.asarray(vocal).reshape(-1).astype(np.float32);n=len(voice)
    crest=float(np.percentile(abs(voice),99.9)/max(np.sqrt(np.mean(voice**2)),1e-9))
    ratio=float(np.clip(1.3+(crest/max(settings['source_crest_factor'],1)-1)*.8,1.2,2.5))
    # Mild dynamics control; reference EQ is deliberately not copied across
    # singer formants, which could pull the timbre away from Voice.
    board=Pedalboard([HighpassFilter(cutoff_frequency_hz=65),Compressor(threshold_db=-20,ratio=ratio,attack_ms=12,release_ms=100)])
    lead=board(voice,sr).reshape(-1)
    stereo=np.repeat(lead[:,None],2,axis=1)
    # Calibrate the installed reverb against its own impulse decay. Room-size
    # controls are not seconds, so a fixed conversion would not match tails.
    room_candidates=[]
    impulse=np.zeros((2,round(7*sr)),dtype=np.float32);impulse[:,0]=1
    for room_value in np.linspace(.2,.9,8):
        response=Reverb(room_size=float(room_value),damping=.55,wet_level=1,dry_level=0,width=1)(impulse.copy(),sr)
        energy=np.cumsum(np.sum(response.astype(float)**2,axis=0)[::-1])[::-1]
        db=10*np.log10(np.maximum(energy/max(float(energy[0]),1e-12),1e-12))
        ids=np.where((db<-5)&(db>-35))[0]
        if len(ids)<20:continue
        slope=float(np.polyfit(ids/sr,db[ids],1)[0])
        if slope<0:room_candidates.append((float(room_value),-60/slope))
    if not room_candidates:raise ValueError('Could not calibrate reverb decay')
    room,calibrated_rt60=min(room_candidates,key=lambda q:abs(np.log(q[1]/settings['reference_rt60_estimate_seconds'])))
    rev=Pedalboard([Reverb(room_size=room,damping=.55,wet_level=1,dry_level=0,width=1),HighpassFilter(cutoff_frequency_hz=220),LowpassFilter(cutoff_frequency_hz=7500)])
    padded=np.pad(stereo,((round(.024*sr),round(4*sr)),(0,0)))
    ambience=rev(padded.T.copy(),sr).T[:n]
    # Duck ambience while a clear lead is present; release through phrase gaps.
    rms=np.sqrt(np.maximum(uniform_filter1d(lead.astype(float)**2,size=round(.04*sr)),0))
    activity=np.clip(rms/max(float(np.percentile(rms,95))*.25,1e-8),0,1)
    duck=gaussian_filter1d(1-.4*activity,sigma=.02*sr)
    ambience*=duck[:,None]
    send=settings['reverb_send'];wet_rms=np.sqrt(np.mean(ambience**2));lead_rms=np.sqrt(np.mean(lead**2))
    ambience*=send*lead_rms/max(wet_rms,1e-9)
    doubles=np.zeros_like(stereo)
    for channel,delay in enumerate([.013,.021]):
        k=round(delay*sr);doubles[k:,channel]=lead[:-k]
    doubles=sosfiltfilt(butter(2,250,fs=sr,btype='highpass',output='sos'),doubles,axis=0)
    double_gain=float(np.clip(settings['stereo_side_mid_ratio']*.65,0,.09))
    echo=np.zeros_like(stereo)
    if settings['delay_seconds']:
        filtered=sosfiltfilt(butter(2,5500,fs=sr,output='sos'),lead)
        for ch in range(2):
            for tap in range(1,4):
                k=round((settings['delay_seconds']*tap+ch*.009)*sr)
                if k<n:echo[k:,ch]+=filtered[:-k]*settings['delay_send']*.35**(tap-1)
    processed=stereo+ambience+doubles*double_gain+echo
    if len(processed)!=n or not np.isfinite(processed).all():raise ValueError('Invalid production output')
    settings.update(compressor_ratio=ratio,reverb_room_parameter=room,calibrated_reverb_rt60_seconds=calibrated_rt60,generated_double_gain=double_gain,
        generated_double_offsets_ms=[13,21],reverb_predelay_ms=24,lead_timeline_unchanged=True,
        effect_audio_origin='All reverb, delay and doubles are derived from the generated Voice vocal. Source waveforms are used for analysis only.',
        tail_policy='Effect tails rendered internally; export retains exact song duration.')
    return processed,settings
