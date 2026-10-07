"""Lightweight dispatcher; --help and init do not import a GPU backend."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from .common import load_profile,write_json,APPLIO_COMMIT

def main():
    p=argparse.ArgumentParser(prog='voice-cover',description='Prepare your singing references, train a voice, and generate covers.')
    p.add_argument('command',choices=['init','doctor','prepare','train','generate','produce'])
    args,remaining=p.parse_known_args()
    if args.command=='init':
        q=argparse.ArgumentParser(prog='voice-cover init');q.add_argument('--out',default='voice.json');q.add_argument('--applio',default='tools/Applio');q.add_argument('--runtime-python',default=sys.executable);q.add_argument('--separator-python',default=sys.executable);q.add_argument('--ffmpeg',default=shutil.which('ffmpeg') or 'ffmpeg');q.add_argument('--model',default='models/voice.pth');q.add_argument('--index',default='models/voice.index');a=q.parse_args(remaining)
        dst=Path(a.out).resolve()
        if dst.exists():q.error('Profile exists; use a new path or edit the existing file')
        cfg=json.loads(Path(__file__).with_name('default_profile.json').read_text())
        for key in ['applio','runtime_python','separator_python','model','index']:
            cfg[key]=str(Path(getattr(a,key)).resolve())
        cfg['ffmpeg']=str(Path(a.ffmpeg).resolve()) if Path(a.ffmpeg).exists() else a.ffmpeg
        cfg['separator_models']=str(dst.parent/'state/separator-models');write_json(dst,cfg);print(dst);return
    if args.command=='doctor':
        q=argparse.ArgumentParser(prog='voice-cover doctor');q.add_argument('--profile',default='voice.json');q.add_argument('--training',action='store_true');a=q.parse_args(remaining);cfg=load_profile(a.profile);checks=[]
        keys=['runtime_python','separator_python','ffmpeg','applio']+([] if a.training else ['model','index'])
        for key in keys:checks.append(dict(item=key,ok=Path(cfg[key]).exists(),path=cfg[key]))
        backend=Path(cfg['applio']);result=subprocess.run(['git','-C',str(backend),'rev-parse','HEAD'],capture_output=True,text=True)
        checks.append(dict(item='pinned Applio revision',ok=result.returncode==0 and result.stdout.strip()==APPLIO_COMMIT,expected=APPLIO_COMMIT))
        resources=['predictors/rmvpe.pt','embedders/contentvec/config.json','embedders/contentvec/pytorch_model.bin']
        if a.training:resources+=['pretraineds/hifi-gan/f0G40k.pth','pretraineds/hifi-gan/f0D40k.pth']
        for asset in resources:checks.append(dict(item=asset,ok=(backend/'rvc/models'/asset).is_file()))
        code='import torch,librosa,parselmouth,swift_f0,pedalboard,pyloudnorm; print(torch.cuda.is_available())'
        runtime=subprocess.run([cfg['runtime_python'],'-c',code],capture_output=True,text=True)
        checks.append(dict(item='audio runtime imports',ok=runtime.returncode==0,detail=runtime.stderr[-1200:] if runtime.returncode else runtime.stdout.strip()))
        sep=subprocess.run([cfg['separator_python'],'-c','from audio_separator.separator import Separator'],capture_output=True,text=True)
        checks.append(dict(item='separator runtime import',ok=sep.returncode==0,detail=sep.stderr[-1200:] if sep.returncode else 'available'))
        print(json.dumps(checks,indent=2));raise SystemExit(0 if all(c['ok'] for c in checks) else 1)
    modules={'prepare':'prepare','train':'train','generate':'generate','produce':'production'}
    # Generate/pitch modules support reexecution in the configured runtime.
    sys.argv=[sys.argv[0],*remaining]
    import runpy
    runpy.run_module('voice_cover.'+modules[args.command],run_name='__main__')
