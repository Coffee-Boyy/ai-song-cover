"""Train a new RVC voice from a reviewed, recording-disjoint dataset."""
import argparse,json,os,re,shutil,subprocess,sys
from pathlib import Path
from .common import load_profile,write_json,ensure_backend_config

def training_commands(cfg,experiment,dataset,name,epochs,batch_size,gpu):
    root=Path(cfg['applio']);py=cfg['runtime_python'];pre=root/'rvc/models/pretraineds/hifi-gan'
    return [
        [py,'rvc/train/preprocess/preprocess.py',str(experiment),str(dataset),'40000','2','Skip','False','False','0.0','3.0','0.3','none'],
        [py,'rvc/train/extract/extract.py',str(experiment),'rmvpe','2',gpu,'40000','contentvec','None','0'],
        [py,str(Path(__file__).with_name('training_bootstrap.py')),name,str(min(20,epochs)),str(epochs),str(pre/'f0G40k.pth'),str(pre/'f0D40k.pth'),gpu,str(batch_size),'40000','False','True','False','False','HiFi-GAN','False'],
        [py,'rvc/train/process/extract_index.py',str(experiment),'Faiss'],
    ]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--name',required=True);p.add_argument('--profile',default='voice.json');p.add_argument('--out-profile',required=True);p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--gpu',default='0');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if not re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,63}',a.name):p.error('Use a simple unique model name without path characters')
    if a.epochs<1 or a.batch_size<1:p.error('Epoch and batch counts must be positive')
    cfg=load_profile(a.profile);manifest_path=Path(a.manifest).resolve();manifest=json.loads(manifest_path.read_text())
    if not manifest.get('approved'):p.error('Review singing intervals and approve manifest.json before training')
    rows=[r for r in manifest['clips'] if r.get('enabled',True) and r['split']=='train'];validation=[r for r in manifest['clips'] if r.get('enabled',True) and r['split']=='validation']
    if not rows:p.error('No enabled training clips')
    if len(rows)<3:p.error('Upstream training requires at least three TRAIN clips; supply additional singing material')
    effective_batch_size=min(a.batch_size,max(1,len(rows)//3))
    if {r['source_sha256'] for r in rows}&{r['source_sha256'] for r in validation}:p.error('Training and validation recordings overlap')
    if len({r['file'] for r in rows})!=len(rows):p.error('Duplicate training files')
    backend=Path(cfg['applio']);experiment=backend/'logs'/a.name;stage=manifest_path.parent/(a.name+'-train-only');out_profile=Path(a.out_profile).resolve()
    if experiment.exists() or stage.exists() or out_profile.exists():p.error('Experiment, staging directory or output profile exists; use a fresh name')
    commands=training_commands(cfg,experiment,stage,a.name,a.epochs,effective_batch_size,a.gpu)
    if a.plan_only:print(json.dumps(dict(commands=commands,train_clips=len(rows),validation_clips=len(validation)),indent=2));return
    pre=backend/'rvc/models/pretraineds/hifi-gan'
    for path in [pre/'f0G40k.pth',pre/'f0D40k.pth']:
        if not path.is_file():raise FileNotFoundError(f'Missing generic pretraining weight: {path}; run setup_backend.py --download-assets')
    ensure_backend_config(backend)
    stage.mkdir();
    for i,r in enumerate(rows):
        file=Path(r['file']);file=file if file.is_absolute() else manifest_path.parent/file
        shutil.copy2(file,stage/f'{i:04d}.wav')
    for i,command in enumerate(commands):
        subprocess.run(command,cwd=backend,check=True)
        if i==0 and not list((experiment/'sliced_audios').glob('*.wav')):raise RuntimeError('Preprocessing produced no training audio')
        if i==1 and not (experiment/'filelist.txt').is_file():raise RuntimeError('Extraction produced no feature filelist')
        if i==2 and not list(experiment.glob(a.name+'_*e_*s.pth')):raise RuntimeError('Training did not export inference weights; no index/profile will be presented as trained')
    checkpoints=list(experiment.glob(a.name+'_*e_*s.pth'))
    if not checkpoints:raise RuntimeError('Training did not export inference weights')
    checkpoint=max(checkpoints,key=lambda p:p.stat().st_mtime_ns);index=experiment/(a.name+'.index')
    if not index.is_file():raise RuntimeError('No retrieval index was exported')
    cfg['name']=a.name;cfg['model']=str(checkpoint);cfg['index']=str(index)
    # Calibrate the working register from TRAIN pitch only; leave heldout data
    # out of the index and range calibration. Limits remain user-editable.
    import numpy as np
    tracks=[np.load(path) for path in (experiment/'f0_voiced').glob('*.npy')]
    voiced=np.concatenate([v[v>0] for v in tracks]) if tracks else np.array([])
    if len(voiced)>20:
        q1,q10,q50,q95,q99=np.percentile(voiced,[1,10,50,95,99]);policy=cfg['register_policy']
        policy.update(center_hz=float(q50),floor_median_hz=float(max(65,q10*.8)),floor_p10_hz=float(max(50,q1*.8)),ceiling_p90_hz=float(min(950,max(q50,q95*1.15))),ceiling_p99_hz=float(min(1040,max(q50,q99*1.1))))
    cfg['calibration']='Empirical limits derived from nonzero TRAIN RMVPE pitch labels; not physiological limits or a quality guarantee.'
    write_json(out_profile,cfg);write_json(manifest_path.parent/(a.name+'-training-report.json'),dict(profile=str(out_profile),commands=commands,model=str(checkpoint),index=str(index),train_clips=len(rows),heldout_clips=len(validation),heldout_in_index=False,epochs=a.epochs,limitations='Training smoke checks do not establish identity quality. Evaluate the recording-disjoint heldout material and listen to new covers.'))
    print(json.dumps(dict(profile=str(out_profile),model=str(checkpoint),index=str(index))))

if __name__=='__main__':main()
