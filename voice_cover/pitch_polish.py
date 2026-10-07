"""Conservative source-guided pitch correction of an existing Voice render.

Only short, sustained, independently measurable pitch deviations are candidates.
No nearest-note quantization, new voice training, timing stretch or EQ is used.
"""
import argparse,hashlib,html,json,os,subprocess,sys,tempfile
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,soundfile as sf,parselmouth
from parselmouth.praat import call
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--native-only',action='store_true',help='Pipeline integration: write corrected native vocal and audit without mixing/export')
    a=p.parse_args();src=Path(a.render_dir).resolve();out=Path(a.out).resolve()
    r=json.loads((src/'report.json').read_text(encoding='utf8'));cfg=r['profile']
    out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']='0';sys.path.insert(0,cfg['applio']);os.chdir(cfg['applio'])
    import torch,librosa
    from voice_cover.f0 import PitchTracker
    torch.set_num_threads(4)
    tracker=PitchTracker('cuda:0' if torch.cuda.is_available() else 'cpu')
    y,sr=sf.read(src/'native-vocal.wav');assert y.ndim==1 and np.isfinite(y).all()
    source=np.load(src/'source-f0.npy');generated=np.load(src/'output-f0.npy');n=min(len(source),len(generated))
    source=source[:n];generated=generated[:n];times=np.arange(n)*.01
    ends=np.array([q['end'] for q in r['plan']]);ids=np.clip(np.searchsorted(ends,times,side='right'),0,len(ends)-1)
    offsets=np.array([q['semitones'] for q in r['plan']])[ids];target=source*2**(offsets/12)
    valid=(target>0)&(generated>0);error=np.zeros(n)
    error[valid]=1200*np.log2(target[valid]/generated[valid])
    smoothed=median_filter(error,size=7)
    # Small errors remain natural. Large errors need re-rendering, not blind retuning.
    mask=valid&(np.abs(smoothed)>45)&(np.abs(smoothed)<180)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    candidates=[(int(i),int(j)) for i,j in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if j-i>=12]
    # Brief large pitch excursions can create chirps even when phrase medians
    # look correct. They still require independent pulse analysis and proof of
    # improvement; sustained large errors remain generation-review failures.
    glitch=valid&(abs(median_filter(error,size=3))>=180)&(abs(median_filter(error,size=3))<=1300)
    edges=np.diff(np.r_[False,glitch,False].astype(int))
    candidates.extend((int(i),int(j)) for i,j in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if 5<=j-i<=25)
    candidates.sort()
    corrected=y.copy();events=[];applied_mask=np.zeros(len(y),dtype=bool)
    for start,end in candidates:
        t0,t1=start*.01,end*.01;shift=float(np.median(error[start:end]))
        # One-second context supplies glottal pulses; only the event is patched.
        aa=max(0,round((t0-1)*sr));bb=min(len(y),round((t1+1)*sr));clip=y[aa:bb]
        snd=parselmouth.Sound(clip,sampling_frequency=sr)
        manip=call(snd,'To Manipulation',.01,65,1100);tier=call(manip,'Extract pitch tier')
        points=[(float(call(tier,'Get time from index',i)),float(call(tier,'Get value at index',i))) for i in range(1,int(call(tier,'Get number of points'))+1)]
        independent=[]
        for t,f in points:
            k=round((t+aa/sr)*100)
            if start<=k<end and generated[k]>0:independent.append(abs(1200*np.log2(f/generated[k])))
        event=dict(start=t0,end=t1,correction_cents=shift)
        if len(independent)<5 or float(np.median(independent))>80:
            event.update(applied=False,reason='Independent pitch tracker disagreed');events.append(event);continue
        new=call('Create PitchTier','source_guided',0,len(clip)/sr)
        def weight(t):
            if t<t0 or t>t1:return 0.
            fade=min(.03,(t1-t0)/4)
            return min(1.,.5-.5*np.cos(np.pi*min((t-t0)/fade,1.)),.5-.5*np.cos(np.pi*min((t1-t)/fade,1.)))
        for t,f in points:call(new,'Add point',t,float(f*2**(shift*weight(t+aa/sr)/1200)))
        call([new,manip],'Replace pitch tier');resynth=call(manip,'Get resynthesis (overlap-add)').values[0]
        if len(resynth)!=len(clip):raise ValueError('PSOLA duration changed')
        weights=np.array([weight(t+aa/sr) for t in np.arange(len(clip))/sr])
        trial=clip*(1-weights)+resynth*weights
        f0=tracker.get_f0(librosa.resample(trial,orig_sr=sr,target_sr=16000,res_type='soxr_vhq').astype(np.float32),.03)
        # Compare to the intended melody, never rank this against old voice generations.
        indices=np.arange(start,end);local=np.round((indices*.01-aa/sr)*100).astype(int)
        good=(local>=0)&(local<len(f0));indices=indices[good];local=local[good]
        common=(target[indices]>0)&(generated[indices]>0)&(f0[local]>0)
        before=np.abs(1200*np.log2(generated[indices[common]]/target[indices[common]]))
        after=np.abs(1200*np.log2(f0[local[common]]/target[indices[common]]))
        retained=float(np.mean(f0[local]>0));peak=float(abs(trial).max())
        event.update(median_pitch_error_before_cents=float(np.median(before)) if len(before) else None,median_pitch_error_after_cents=float(np.median(after)) if len(after) else None,voiced_retention=retained)
        accepted=len(after)>=5 and np.median(after)<np.median(before)*.85 and retained>=.9 and np.percentile(after,90)<=np.percentile(before,90)+10 and peak<.97
        event.update(applied=bool(accepted),reason='Melody error reduced with voicing retained' if accepted else 'Did not pass the melody improvement guard')
        if accepted:
            region=corrected[aa:bb];region[weights>0]=trial[weights>0];applied_mask[aa:bb]|=weights>0
        events.append(event)
    assert np.array_equal(y[~applied_mask],corrected[~applied_mask])
    assert len(y)==len(corrected) and np.isfinite(corrected).all() and abs(corrected).max()<1
    sf.write(out/'postprocessed-native-vocal.wav',corrected,sr,subtype='FLOAT')
    report=dict(method='Source-guided local TD-PSOLA; constant pitch correction per event with raised-cosine onset/release; original waveform outside accepted events.',input_render=str(src),input_model_sha256=r['model_sha256'],input_native_sha256=hashlib.sha256((src/'native-vocal.wav').read_bytes()).hexdigest(),candidate_events=events,accepted_events=sum(e['applied'] for e in events),changed_seconds=float(applied_mask.sum()/sr),duration_seconds=len(y)/sr,untouched_samples_identical=True,quality_scope='Pitch-only proof of concept. Does not reconstruct Voice timbre, fix pronunciation, or certify perceptual quality. Extreme pitch/voicing failures require regeneration.',references=['https://www.fon.hum.uva.nl/praat/manual/Manipulation.html','https://www.fon.hum.uva.nl/praat/manual/Intro_8_1__Manipulation_of_pitch.html'])
    if a.native_only:
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps({'accepted_events':report['accepted_events'],'changed_seconds':report['changed_seconds'],'native_vocal':str(out/'postprocessed-native-vocal.wav')}));return
    export_sr=44100
    from math import gcd
    def resample(x,s):
        g=gcd(export_sr,s);return resample_poly(x,export_sr//g,s//g,axis=0) if s!=export_sr else x
    voice=resample(corrected,sr)[:,None];length=round(r['duration_seconds']*export_sr)
    voice=np.pad(voice,((0,max(0,length-len(voice))),(0,0)))[:length]
    import pyloudnorm as pyln
    meter=pyln.Meter(export_sr)
    def level(x,target):
        l=float(meter.integrated_loudness(x));assert np.isfinite(l)
        return x*min(10**((target-l)/20),.96/max(float(abs(x).max()),1e-9))
    audio=[]
    def export(x,label,loudness):
        x=level(x,loudness);wav=out/(label+'.wav');mp3=wav.with_suffix('.mp3');gain=1
        for _ in range(8):
            sf.write(wav,x*gain,export_sr,subtype='PCM_24');subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',str(wav),'-c:a','libmp3lame','-b:a','320k',str(mp3)],check=True)
            with tempfile.TemporaryDirectory() as td:
                d=Path(td)/'decoded.wav';subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',str(mp3),'-c:a','pcm_f32le',str(d)],check=True);z,_=sf.read(d)
            peak=float(abs(z).max());assert np.isfinite(z).all()
            if peak<.97:break
            gain*=.958/peak
        else:raise ValueError('MP3 clipping guard failed')
        audio.append(dict(label=label,wav=str(wav),mp3=str(mp3),decoded_mp3_peak=peak))
        return '<h2>'+html.escape(label)+'</h2><audio controls src="'+label+'.mp3"></audio><p><a download href="'+label+'.wav">WAV</a> · <a download href="'+label+'.mp3">MP3</a></p>'
    backing_path=Path(r.get('song',{}).get('separation_report',''))
    if r.get('backing'):backing_path=Path(r['backing'])
    elif backing_path.is_file():backing_path=Path(json.loads(backing_path.read_text())['backing'])
    elif (src/'stems/backing.wav').is_file():backing_path=src/'stems/backing.wav'
    else:backing_path=None
    players=''
    if backing_path:
        b,bsr=sf.read(backing_path,always_2d=True);b=resample(b,bsr);assert abs(len(b)-length)/export_sr<.05
        b=np.pad(b,((0,max(0,length-len(b))),(0,0)))[:length];v=level(voice,-18);b=level(b,-20)
        if b.shape[1]>1:v=np.repeat(v,b.shape[1],axis=1)
        players+=export(v+b,'cover',-16)
    players+=export(voice,'vocal',-18)
    report['audio']=audio;(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    title='Voice · '+r.get('song',{}).get('title','cover')+' · pitch polish'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{background:#101722;color:#eef3ff;font:17px system-ui;max-width:940px;margin:auto;padding:30px}a{color:#accfff}audio{width:100%}</style><h1>'+html.escape(title)+'</h1><p>Conservative source-guided correction. '+str(report['accepted_events'])+' accepted spots, '+str(round(report['changed_seconds'],2))+' seconds changed. Timing and audio outside those spots are unchanged. Pitch correction does not repair voice identity or pronunciation.</p>'+players+'<p><a href="report.json">Processing report</a></p><script>document.querySelectorAll("audio").forEach(a=>a.onplay=()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()}));</script>',encoding='utf8')
    print(json.dumps({'page':str(out/'index.html'),'report':report},indent=2))

if __name__=='__main__':main()
