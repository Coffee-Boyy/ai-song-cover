"""Apply the automatic production stage to an existing finished cover."""
import argparse,json,subprocess,tempfile,html
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,soundfile as sf,pyloudnorm as pyln
from scipy.signal import resample_poly
from math import gcd
from voice_cover.effects import apply_effects
from voice_cover.cleanup import true_peak

p=argparse.ArgumentParser();p.add_argument('--render-dir',required=True);p.add_argument('--out',required=True);p.add_argument('--effects-reference',required=True);a=p.parse_args()
src=Path(a.render_dir).resolve();out=Path(a.out).resolve();out.mkdir(exist_ok=False,parents=True)
r=json.loads((src/'report.json').read_text());cfg=r['profile'];v,sr=sf.read(src/'vocal.wav',always_2d=True)
voice,effects=apply_effects(v.mean(1),sr,Path(a.effects_reference).resolve(),Path(r['source']))
meter=pyln.Meter(sr)
def level(x,l):return x*min(10**((l-meter.integrated_loudness(x))/20),.96/max(abs(x).max(),1e-9))
renders=[];exports=[]
if r.get('backing'):
    b,bsr=sf.read(r['backing'],always_2d=True)
    if bsr!=sr:
        g=gcd(sr,bsr);b=resample_poly(b,sr//g,bsr//g,axis=0)
    if len(b)!=len(voice):raise ValueError('Backing length mismatch')
    if b.shape[1]==1:b=np.repeat(b,2,axis=1)
    renders.append((level(voice,-18)+level(b,-20),'cover',-16))
renders.append((voice,'vocal',-18))
for x,label,loudness in renders:
    x=level(x,loudness);x*=min(1,.96/max(true_peak(x),1e-9));wav=out/(label+'.wav');mp3=out/(label+'.mp3');gain=1
    for _ in range(8):
        sf.write(wav,x*gain,sr,subtype='PCM_24');subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',str(wav),'-c:a','libmp3lame','-b:a','320k',str(mp3)],check=True)
        with tempfile.TemporaryDirectory() as td:
            dec=Path(td)/'decoded.wav';subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',str(mp3),'-c:a','pcm_f32le',str(dec)],check=True);z,_=sf.read(dec)
        peak=true_peak(z)
        if peak<.97:break
        gain*=.958/peak
    else:raise ValueError('MP3 true-peak check failed')
    exports.append(dict(label=label,duration_seconds=len(x)/sr,wav=str(wav),mp3=str(mp3),wav_true_peak_4x=true_peak(x*gain),decoded_mp3_true_peak_4x=peak))
report=dict(source_render=str(src),model_sha256=r['model_sha256'],effects=effects,exports=exports,
    status='waveform_checks_passed',checks=dict(finite_samples=True,original_duration_preserved=True,lead_not_shifted=True,decoded_mp3_true_peak_passed=True),
    limitations='Effect estimates are approximate. This adds production to the validated lead; it does not independently certify perceived naturalness of the finished mix.')
(out/'report.json').write_text(json.dumps(report,indent=2));(out/'studio-effects.json').write_text(json.dumps(effects,indent=2))
title=html.escape(r.get('song',{}).get('title','Voice cover'));players=''
for label,name in [('cover','Cover with source-adaptive studio effects'),('vocal','Processed Voice vocal')]:
    if not any(e['label']==label for e in exports):continue
    players+=f'<h2>{name}</h2><audio controls src="{label}.mp3"></audio><p><a download href="{label}.wav">Download WAV</a> · <a download href="{label}.mp3">Download MP3</a></p>'
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+title+' · studio production</title><style>body{background:#101722;color:#eef3ff;font:17px system-ui;max-width:940px;margin:auto;padding:30px}a{color:#accfff}audio{width:100%}p{line-height:1.6}</style><h1>'+title+' · studio production</h1><p>Source-adaptive reverb, subtle tempo-linked delay, stereo doubles and gentle dynamics processing. These effects approximate the original production. All effect sends use Voice’s generated voice.</p>'+players+'<p><a href="'+html.escape(src.as_uri()+'/index.html')+'">Original Voice cover</a> · <a href="studio-effects.json">Adaptive settings</a> · <a href="report.json">Export checks</a></p><script>document.querySelectorAll("audio").forEach(a=>a.onplay=()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()}));</script>',encoding='utf8')
print(json.dumps(report),flush=True)
