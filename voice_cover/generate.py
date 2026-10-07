"""Voice cover CLI: phrase-level register control in ONE continuous RVC render."""
import argparse,hashlib,html,json,os,subprocess,sys,tempfile,uuid
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_cover.common import load_profile
from voice_cover.backend import overrides
from voice_cover.register import plan_register,phrase_metrics,register_ramps,ramp_shift,clean_pitch_track,retry_register_allowed

WORKSPACE=Path.cwd()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--song',help='Song/video file to separate automatically')
    source.add_argument('--vocals',help='Already isolated lead vocal')
    p.add_argument('--backing',help='Aligned instrumental stem with --vocals')
    p.add_argument('--out',help='Fresh output directory')
    p.add_argument('--profile',default=str(WORKSPACE/'voice.json'))
    p.add_argument('--gpu',default='0',help='PyTorch GPU index, default 0')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--skip-pitch-polish',action='store_true',help='Diagnostic opt-out of automatic guarded pitch correction')
    p.add_argument('--review-preview',action='store_true',help='Export a clearly flagged test preview if final cleanup checks need review')
    p.add_argument('--effects-reference',help='Original wet vocal stem for adaptive studio production; automatically found with --song')
    p.add_argument('--skip-effects',action='store_true',help='Export without the automatic source-adaptive production stage')
    args=p.parse_args();profile=Path(args.profile).resolve();cfg=load_profile(profile)
    runtime=Path(cfg['runtime_python'])
    if Path(sys.executable).resolve()!=runtime.resolve():
        raise SystemExit(subprocess.run([str(runtime),str(Path(__file__).resolve()),*sys.argv[1:]]).returncode)
    if args.song and args.backing:p.error('--backing is only used with --vocals')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    input_path=Path(args.song or args.vocals).resolve()
    backing=Path(args.backing).resolve() if args.backing else None
    effects_reference=Path(args.effects_reference).resolve() if args.effects_reference else None
    out=Path(args.out).resolve() if args.out else WORKSPACE/'outputs'/('voice-cover-'+uuid.uuid4().hex[:8])
    applio=Path(cfg['applio']);sys.path.insert(0,str(applio));os.chdir(applio)
    def resolve(value):
        value=Path(value);return value if value.is_absolute() else WORKSPACE/value
    model,index=resolve(cfg['model']),resolve(cfg['index'])
    for path in [model,index,Path(cfg['ffmpeg'])]:
        if not path.is_file():raise FileNotFoundError(path)
    import numpy as np,torch,soundfile as sf,librosa,pyloudnorm as pyln
    from scipy.signal import resample_poly
    from scipy.ndimage import uniform_filter1d
    from math import gcd
    from rvc.infer.infer import VoiceConverter
    from rvc.infer.pipeline import Pipeline
    from voice_cover.f0 import PitchTracker
    from voice_cover.pitch_guard import repair_source_guide
    from voice_cover.cleanup import cleanup_audio,audit_pitch_islands,true_peak
    torch.set_num_threads(4)
    out.mkdir(parents=True,exist_ok=False)
    if not input_path.is_file():raise FileNotFoundError(input_path)
    if args.song:
        subprocess.run([cfg['separator_python'],str(Path(__file__).with_name('separate.py')),'--song',str(input_path),'--out',str(out/'stems'),'--profile',str(profile)],check=True)
        stems=json.loads((out/'stems/stems.json').read_text());input_path=Path(stems['vocals']);backing=Path(stems['backing'])
        if effects_reference is None:
            wet=list((out/'stems').glob('source-mix_(Vocals)*.wav'))
            effects_reference=wet[0] if len(wet)==1 else input_path
        if cfg.get('studio_effects',{}).get('enabled',True) and not args.skip_effects:
            subprocess.run([cfg['separator_python'],str(Path(__file__).with_name('dereverb.py')),'--vocals',str(input_path),'--out',str(out/'foreground'),'--profile',str(profile)],check=True)
            input_path=out/'foreground/dry-vocals.wav'
    if backing and not backing.is_file():raise FileNotFoundError(backing)
    if effects_reference and not effects_reference.is_file():raise FileNotFoundError(effects_reference)
    # Prepare mono once. No crop, stretch, splice or time warp is used.
    x,sr=sf.read(input_path,always_2d=True)
    if not np.isfinite(x).all():raise ValueError('Nonfinite input samples')
    mono=x.mean(1)
    if np.sqrt(np.mean(mono**2))<.35*np.sqrt(np.mean(x**2)):mono=x[:,np.argmax(np.mean(x**2,axis=0))]
    prepared=out/'source-vocals.wav';sf.write(prepared,mono,sr,subtype='FLOAT');duration=len(mono)/sr
    x16=librosa.resample(mono,orig_sr=sr,target_sr=16000,res_type='soxr_vhq').astype(np.float32)
    f0model=PitchTracker('cuda:0' if torch.cuda.is_available() else 'cpu')
    source_f0=f0model.get_f0(x16,.03)
    source_octave_corrections=f0model.last_corrected_frames
    if cfg.get('source_pitch_method','consensus')=='swift':
        from rvc.lib.predictors.f0 import Swift
        source_f0=Swift(device='cpu').get_f0(x16,f0_max=2500,threshold=.5)
        source_octave_corrections=0
    source_guard=dict(enabled=False)
    if cfg.get('source_pitch_guard',{}).get('enabled',True):
        np.save(out/'source-f0-before-guard.npy',source_f0)
        source_f0,source_guard=repair_source_guide(source_f0,f0model.last_stock,f0model.last_praat,f0model.last_praat_strength)
        np.save(out/'source-stock-f0.npy',f0model.last_stock)
        np.save(out/'source-praat-f0.npy',f0model.last_praat)
        (out/'source-pitch-guard.json').write_text(json.dumps(source_guard,indent=2))
    # Ignore pitch estimates below the source stem's very conservative energy floor.
    # This gates pitch analysis only: no waveform samples or quiet breaths are cut.
    energy=np.sqrt(np.maximum(uniform_filter1d(x16.astype(float)**2,size=800),0))
    frame_energy=energy[np.minimum(np.round(np.arange(len(source_f0))*160).astype(int),len(energy)-1)]
    floor=max(10**(-70/20),float(np.percentile(frame_energy,95))*10**(-40/20))
    quiet_floor=max(floor,float(np.percentile(frame_energy,95))*10**(-30/20))
    cleaned=np.asarray(clean_pitch_track(source_f0,frame_energy,floor,quiet_floor))
    source_rejected=int(np.sum((source_f0>0)&(cleaned==0)))
    source_f0=cleaned
    np.save(out/'source-f0.npy',source_f0)
    if np.sum(source_f0>0)<20:raise ValueError('Not enough detectable singing in source stem')
    plan=plan_register(source_f0,duration,cfg.get('register_policy'))
    # Snap each register join to the lowest 50ms energy inside its unvoiced gap.
    for r in plan[1:]:
        if not r['join_gap']:continue
        a,b=r['join_gap'];centers=np.arange(a+.04,b-.04,.01)
        if len(centers):
            energy=[float(np.mean(x16[max(0,round((t-.025)*16000)):min(len(x16),round((t+.025)*16000))]**2)) for t in centers]
            r['start']=float(centers[int(np.argmin(energy))])
    for i in range(len(plan)-1):plan[i]['end']=plan[i+1]['start']
    (out/'register-plan.json').write_text(json.dumps(plan,indent=2))
    if args.plan_only:print(json.dumps({'register_plan':str(out/'register-plan.json')}));return
    vc=VoiceConverter();original_get_f0=Pipeline.get_f0;attempts=[];tried=[set() for _ in plan]
    def controlled_f0(pipe,audio,p_len,*unused,**unused_kw):
        times=np.arange(p_len)*.01-pipe.x_pad
        # One authoritative pitch contour drives planning, synthesis and auditing.
        # Re-estimating after Applio's filtering/padding can change octave labels
        # in remixed passages. Reflect the measured contour for inference padding.
        positions=np.rint(times*100).astype(int);period=2*(len(source_f0)-1)
        folded=positions%period;indices=np.where(folded<len(source_f0),folded,period-folded)
        base=source_f0[indices]
        ends=np.array([r['end'] for r in plan]);ids=np.clip(np.searchsorted(ends,times,side='right'),0,len(plan)-1)
        shifts=np.array([r['semitones'] for r in plan])[ids]
        shifts=shifts.astype(float)
        # Continuous raised-cosine register ramps within unvoiced phrase gaps.
        # No voiced note is swept across an octave and no waveform is spliced.
        for ramp in register_ramps(plan):
            mask=(times>=ramp['start'])&(times<=ramp['end'])
            shifts[mask]=[ramp_shift(float(t),ramp) for t in times[mask]]
        target=base*2**(shifts/12)
        mel=1127*np.log1p(target/700)
        coarse=np.rint(np.clip((mel-pipe.f0_mel_min)*254/(pipe.f0_mel_max-pipe.f0_mel_min)+1,1,255)).astype(np.int64)
        np.save(out/'conditioning-f0.npy',target.astype(np.float32));np.save(out/'conditioning-register-semitones.npy',shifts)
        return coarse,target.astype(np.float32)
    Pipeline.get_f0=controlled_f0
    raw=out/'native-vocal.wav';success=False
    try:
        for attempt in range(cfg.get('max_guard_retries',2)+1):
            torch.manual_seed(42);np.random.seed(42)
            with overrides(raw,cfg.get("retrieval_backend","exact")):
                vc.convert_audio(str(prepared),str(raw),str(model),str(index),pitch=0,f0_method='rmvpe',index_rate=cfg['inference']['index_rate'],protect=cfg['inference']['protect'],volume_envelope=cfg['inference']['volume_envelope'],embedder_model='contentvec',clean_audio=False,export_format='WAV',post_process=False,f0_autotune=False)
            y,ysr=sf.read(raw)
            if not np.isfinite(y).all() or abs(y).max()>=1:raise ValueError('Invalid or clipped native output')
            y16=librosa.resample(y,orig_sr=ysr,target_sr=16000,res_type='soxr_vhq').astype(np.float32)
            output_f0=f0model.get_f0(y16,.03)
            checks=[phrase_metrics(source_f0,output_f0,r['start'],r['end'],r['semitones']) for r in plan]
            attempts.append(dict(attempt=attempt,registers=[r['semitones'] for r in plan],checks=checks));failed=[i for i,c in enumerate(checks) if not c['passed']]
            if not failed:success=True;break
            if attempt==cfg.get('max_guard_retries',2):break
            changed=False
            for i in failed:
                if (cfg.get('pitch_polish',{}).get('enabled',True) and not args.skip_pitch_polish
                        and not retry_register_allowed(checks[i])):continue
                r=plan[i];tried[i].add(r['semitones']);options=[c for c in r['candidates'] if c['safe'] and c['semitones'] not in tried[i]]
                if options:
                    choice=min(options,key=lambda c:c['cost']+sum(abs(c['semitones']-plan[j]['semitones'])/12 for j in [i-1,i+1] if 0<=j<len(plan)))
                    r['semitones']=choice['semitones'];changed=True
            if not changed:break
    finally:Pipeline.get_f0=original_get_f0
    # Verify joins using voiced/unvoiced timing around each register change.
    timing=[]
    for i,r in enumerate(plan[1:],1):
        if r['semitones']==plan[i-1]['semitones']:continue
        center=round(r['start']*100);a=max(0,center-200);b=min(len(source_f0),len(output_f0),center+200)
        candidates=[]
        for lag in range(-12,13):
            j=np.arange(a,b);k=j+lag;valid=(k>=0)&(k<len(output_f0));j=j[valid];k=k[valid]
            agreement=float(np.mean((source_f0[j]>0)==(output_f0[k]>0)))
            candidates.append((agreement,lag))
        best=min(candidates,key=lambda item:(-item[0],abs(item[1])));zero=next(v for v,l in candidates if l==0)
        flagged=abs(best[1])>8 and best[0]-zero>.05
        timing.append(dict(time=r['start'],estimated_voicing_lag_ms=best[1]*10,zero_lag_agreement=zero,timing_review_required=flagged))
    success=success and not any(t['timing_review_required'] for t in timing)
    report=dict(status='passed' if success else 'needs_review',profile=cfg,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),source=str(input_path),source_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),duration_seconds=duration,plan=plan,attempts=attempts,timing=timing,render_method='Single continuous decode with raised-cosine register ramps inside unvoiced phrase breaks; no register audio splices or time stretching.',limitations='Pitch, voicing and timing diagnostics cannot guarantee exact Voice likeness or naturalness on unseen songs.')
    report['register_ramps']=register_ramps(plan)
    report['conditioning_pitch_source']='Same measured source contour as register planning and quality checks, reflected at inference padding; no second pitch extraction after input filtering.'
    report['source_pitch_method']=cfg.get('source_pitch_method','consensus')
    report['source_pitch_guard']=source_guard
    report['backing']=str(backing) if backing else None
    report['pitch_octave_consensus']=dict(enabled=True,source_corrected_frames=source_octave_corrections,method='Reject high-register octave inflation when stock RMVPE agrees with independent Praat or confidence-gated SwiftF0 within 60 cents; independently supported high notes remain.')
    report['pitch_activity_filter']=dict(source_floor_dbfs=float(20*np.log10(floor)),rejected_source_voiced_frames=source_rejected,window_ms=50,relative_to_source_p95_rms_db=-40,absolute_minimum_dbfs=-70,quiet_island_floor_dbfs=float(20*np.log10(quiet_floor)),quiet_island_max_duration_ms=240,waveform_gating=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'register-plan.json').write_text(json.dumps(plan,indent=2));np.save(out/'output-f0.npy',output_f0)
    pitch_repair_possible=(cfg.get('pitch_polish',{}).get('enabled',True) and not args.skip_pitch_polish and cfg.get('audio_cleanup',{}).get('enabled',True)
        and not any(t['timing_review_required'] for t in timing)
        and all(c.get('skipped') or (c['voiced_retention']>=.8 and c['voicing_agreement']>=.8 and c['median_error_cents'] is not None and c['median_error_cents']<=100) for c in checks))
    report['raw_render_checks_passed']=success
    if not success and not pitch_repair_possible:
        print(json.dumps({'status':'needs_review','report':str(out/'report.json'),'native_preview':str(raw)}));raise SystemExit(2)
    if not success:
        report['status']='pending_pitch_repair';(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    # Correct pitch before the final mix/export; original native render is retained.
    polish_enabled=cfg.get('pitch_polish',{}).get('enabled',True) and not args.skip_pitch_polish
    report['pitch_polish']=dict(enabled=polish_enabled)
    if polish_enabled:
        polish=out/'pitch-polish'
        subprocess.run([str(runtime),str(Path(__file__).with_name('pitch_polish.py')),'--render-dir',str(out),'--out',str(polish),'--native-only'],check=True)
        audit=json.loads((polish/'report.json').read_text(encoding='utf8'))
        polished,polished_sr=sf.read(polish/'postprocessed-native-vocal.wav')
        if polished_sr!=ysr or len(polished)!=len(y) or not np.isfinite(polished).all() or abs(polished).max()>=1:
            raise ValueError('Automatic pitch polish changed duration or produced invalid audio')
        y=polished
        report['pitch_polish'].update(accepted_events=audit['accepted_events'],changed_seconds=audit['changed_seconds'],report=str(polish/'report.json'),native_vocal=str(polish/'postprocessed-native-vocal.wav'))
    if cfg.get('audio_cleanup',{}).get('enabled',True):
        y,cleanup=cleanup_audio(y,ysr,mono,sr)
        sf.write(out/'cleaned-native-vocal.wav',y,ysr,subtype='FLOAT')
        final_f0=f0model.get_f0(librosa.resample(y,orig_sr=ysr,target_sr=16000,res_type='soxr_vhq').astype(np.float32),.03)
        final_checks=[phrase_metrics(source_f0,final_f0,r['start'],r['end'],r['semitones']) for r in plan]
        t=np.arange(len(source_f0))*.01
        ids=np.clip(np.searchsorted([r['end'] for r in plan],t,side='right'),0,len(plan)-1)
        target=source_f0*2**(np.array([r['semitones'] for r in plan])[ids]/12)
        cleanup['unexplained_pitch_islands']=audit_pitch_islands(target,final_f0)
        cleanup['final_phrase_checks']=final_checks
        cleanup['passed']=not cleanup['dropouts_requiring_review'] and not cleanup['unexplained_pitch_islands'] and all(c['passed'] for c in final_checks)
        report['audio_cleanup']=dict(enabled=True,report=str(out/'cleanup-report.json'),passed=cleanup['passed'])
        np.save(out/'final-output-f0.npy',final_f0)
        (out/'cleanup-report.json').write_text(json.dumps(cleanup,indent=2),encoding='utf8')
        if not cleanup['passed']:
            report['status']='needs_review';(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
            if not args.review_preview:
                print(json.dumps({'status':'needs_review','report':str(out/'cleanup-report.json'),'native_preview':str(out/'cleaned-native-vocal.wav')}));raise SystemExit(2)
        else:report['status']='passed'
    else:report['audio_cleanup']=dict(enabled=False)
    export_sr=44100;meter=pyln.Meter(export_sr)
    def resample(a,s):
        d=gcd(s,export_sr);return resample_poly(a,export_sr//d,s//d,axis=0) if s!=export_sr else a
    def level(a,target):
        loudness=float(meter.integrated_loudness(a))
        if not np.isfinite(loudness):raise ValueError('Silent/nonfinite export')
        return a*min(10**((target-loudness)/20),.96/max(float(abs(a).max()),1e-9))
    voice=resample(y,ysr)[:,None];n=round(duration*export_sr)
    if abs(len(voice)-n)/export_sr>.1:raise ValueError('Output duration mismatch')
    voice=np.pad(voice,((0,max(0,n-len(voice))),(0,0)))[:n];audio=[]
    effects_enabled=cfg.get('studio_effects',{}).get('enabled',True) and not args.skip_effects
    report['studio_effects']=dict(enabled=effects_enabled)
    if effects_enabled:
        from voice_cover.effects import apply_effects
        voice,effects=apply_effects(voice,export_sr,effects_reference or prepared,prepared)
        sf.write(out/'produced-vocal.wav',voice,export_sr,subtype='FLOAT')
        effects['reference']=str(effects_reference or prepared)
        (out/'studio-effects.json').write_text(json.dumps(effects,indent=2))
        report['studio_effects'].update(report=str(out/'studio-effects.json'),reference=effects['reference'])
    def export(a,label,target):
        a=level(a,target);a*=min(1,.96/max(true_peak(a),1e-9));wav=out/(label+'.wav');mp3=wav.with_suffix('.mp3');gain=1
        for _ in range(8):
            sf.write(wav,a*gain,export_sr,subtype='PCM_24');subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',str(wav),'-c:a','libmp3lame','-b:a','320k',str(mp3)],check=True)
            with tempfile.TemporaryDirectory() as td:
                decoded=Path(td)/'d.wav';subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',str(mp3),'-c:a','pcm_f32le',str(decoded)],check=True);z,_=sf.read(decoded)
            if not np.isfinite(z).all():raise ValueError('Nonfinite decoded MP3')
            peak=true_peak(z)
            if peak<.97:break
            gain*=.958/peak
        else:raise ValueError('MP3 clipping guard failed')
        audio.append(dict(label=label,wav=str(wav),mp3=str(mp3),duration_seconds=len(a)/export_sr,decoded_mp3_true_peak_4x=peak))
        return f'<h2>{html.escape(label)}</h2><audio controls src="{label}.mp3"></audio><p><a download href="{label}.wav">WAV</a> · <a download href="{label}.mp3">MP3</a></p>'
    players=''
    if backing:
        b,bsr=sf.read(backing,always_2d=True);b=resample(b,bsr)
        if not np.isfinite(b).all() or abs(len(b)-n)/export_sr>.05:raise ValueError('Backing must be finite and aligned to the full vocal stem')
        b=np.pad(b,((0,max(0,n-len(b))),(0,0)))[:n];v=level(voice,-18);b=level(b,-20)
        if v.shape[1]==1 and b.shape[1]>1:v=np.repeat(v,b.shape[1],axis=1)
        players+=export(v+b,'cover',-16)
    players+=export(voice,'vocal',-18);report['audio']=audio
    (out/'report.json').write_text(json.dumps(report,indent=2))
    polish_note=('<p>Automatic source-guided pitch polish: '+str(report['pitch_polish']['accepted_events'])+' accepted corrections. <a href="pitch-polish/report.json">Pitch processing report</a></p>') if polish_enabled else ''
    if effects_enabled:polish_note+='<p>Source-adaptive vocal ambience, dynamics and stereo layering applied. <a href="studio-effects.json">Studio production report</a></p>'
    if report['audio_cleanup']['enabled']:
        quality_text='Whole-track cleanup and final melody/dropout checks passed.' if report['audio_cleanup']['passed'] else 'Review preview: waveform cleanup completed, but final melody checks require review. This is not a validated final cover.'
        polish_note+='<p>'+quality_text+' <a href="cleanup-report.json">Cleanup report</a></p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Voice automatic cover</title><style>body{background:#101722;color:#eef3ff;font:17px system-ui;max-width:940px;margin:auto;padding:30px}a{color:#accfff}audio{width:100%}p{line-height:1.6}</style><h1>Voice · automatic cover</h1><p>Single continuous vocal render. Register changes are planned at source phrase breaks; no audio splicing or time stretching.</p>'+polish_note+players+'<p><a href="report.json">Generation and validation report</a></p><script>document.querySelectorAll("audio").forEach(a=>a.onplay=()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()}));</script>',encoding='utf8')
    print(json.dumps({'status':report['status'],'output':str(out),'page':str(out/'index.html'),'audio':audio},indent=2))

if __name__=='__main__':main()
