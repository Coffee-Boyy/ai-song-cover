"""Prepare aligned source stems using the existing Mel-band separation runtime."""
import argparse,json,os,subprocess
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_cover.common import load_profile
import numpy as np,soundfile as sf
from scipy.signal import butter,sosfiltfilt
from audio_separator.separator import Separator

def main():
    p=argparse.ArgumentParser();p.add_argument('--song',required=True);p.add_argument('--out',required=True);p.add_argument('--profile',required=True);a=p.parse_args()
    cfg=load_profile(a.profile);out=Path(a.out);out.mkdir(exist_ok=True)
    os.environ['PATH']=str(Path(cfg['ffmpeg']).parent)+os.pathsep+os.environ['PATH']
    mix=out/'source-mix.wav'
    subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',a.song,'-vn','-ar','44100','-c:a','pcm_f32le',str(mix)],check=True)
    sep=Separator(output_dir=str(out),output_format='WAV',model_file_dir=cfg['separator_models'],use_soundfile=True,use_autocast=True,mdxc_params={'segment_size':256,'override_model_segment_size':True,'batch_size':1,'overlap':8})
    sep.load_model(cfg['separator_model']);files=sep.separate(str(mix));result={}
    for stem,name in [('Vocals','vocals'),('Instrumental','backing')]:
        paths=[f for f in files if '('+stem+')' in Path(f).name]
        if len(paths)!=1:raise RuntimeError('Expected one '+stem+' stem')
        source=Path(paths[0]);source=source if source.is_absolute() else out/source
        x,sr=sf.read(source,always_2d=True)
        if not np.isfinite(x).all():raise RuntimeError('Nonfinite separated stem')
        if name=='vocals':
            mono=x.mean(1)
            if np.sqrt(np.mean(mono**2))<.35*np.sqrt(np.mean(x**2)):mono=x[:,np.argmax(np.mean(x**2,axis=0))]
            x=sosfiltfilt(butter(2,45,fs=sr,btype='highpass',output='sos'),mono)
            x*=min(1,.95/max(float(abs(x).max()),1e-9))
        dest=out/(name+'.wav');sf.write(dest,x,sr,subtype='FLOAT');result[name]=str(dest.resolve())
    (out/'stems.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
