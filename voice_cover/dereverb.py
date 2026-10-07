"""Extract an aligned foreground singing stem with the installed UVR model."""
import argparse,json,os
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_cover.common import load_profile
import numpy as np,soundfile as sf
from audio_separator.separator import Separator

p=argparse.ArgumentParser();p.add_argument('--vocals',required=True);p.add_argument('--out',required=True);p.add_argument('--profile',required=True);a=p.parse_args()
cfg=load_profile(a.profile);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
os.environ['PATH']=str(Path(cfg['ffmpeg']).parent)+os.pathsep+os.environ['PATH']
sep=Separator(output_dir=str(out),output_format='WAV',model_file_dir=cfg['separator_models'],use_soundfile=True,use_autocast=True,vr_params={'window_size':512,'batch_size':4,'aggression':5,'enable_tta':True})
sep.load_model('UVR-DeEcho-DeReverb.pth');files=sep.separate(str(Path(a.vocals).resolve()))
paths=[f for f in files if '(No Reverb)' in f]
if len(paths)!=1:raise ValueError('Expected foreground stem')
path=Path(paths[0]);path=path if path.is_absolute() else out/path
x,sr=sf.read(path,always_2d=True);original,original_sr=sf.read(a.vocals,always_2d=True)
tail_difference=len(original)-len(x)
if original_sr!=sr or abs(tail_difference)>round(.02*sr) or not np.isfinite(x).all():raise ValueError('Invalid alignment or samples')
# UVR rounds its STFT output length. Restore only the tiny final tail, never
# shift or stretch the foreground singing timeline.
x=np.pad(x,((0,max(0,tail_difference)),(0,0)))[:len(original)]
mono=x.mean(1)
if np.sqrt(np.mean(mono**2))<.35*np.sqrt(np.mean(x**2)):mono=x[:,np.argmax(np.mean(x**2,axis=0))]
sf.write(out/'dry-vocals.wav',mono,sr,subtype='FLOAT')
(out/'report.json').write_text(json.dumps(dict(input=str(Path(a.vocals).resolve()),output=str(out/'dry-vocals.wav'),model='UVR-DeEcho-DeReverb.pth',duration_seconds=len(mono)/sr,alignment_preserved=True,files=files),indent=2))
