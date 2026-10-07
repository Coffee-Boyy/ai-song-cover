"""Fetch the pinned upstream backend and optional generic (non-speaker) assets."""
import argparse,hashlib,json,subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

REVISION='21392273e01da7e0169c18567d60c9b9d40282dc'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='tools/Applio');p.add_argument('--download-assets',action='store_true');a=p.parse_args();root=Path(a.out).resolve()
if not root.exists():subprocess.run(['git','clone','--no-checkout','https://github.com/IAHispano/Applio.git',str(root)],check=True)
current=subprocess.run(['git','-C',str(root),'status','--porcelain'],capture_output=True,text=True,check=True)
if current.stdout.strip():raise SystemExit('Backend has local changes; refusing to change its revision')
subprocess.run(['git','-C',str(root),'checkout','--detach',REVISION],check=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_cover.common import ensure_backend_config
ensure_backend_config(root)
if a.download_assets:
    base='https://huggingface.co/IAHispano/Applio/resolve/main/Resources/'
    files={'predictors/rmvpe.pt':'predictors/rmvpe.pt','embedders/contentvec/config.json':'embedders/contentvec/config.json','embedders/contentvec/pytorch_model.bin':'embedders/contentvec/pytorch_model.bin','pretrained_v2/f0G40k.pth':'pretraineds/hifi-gan/f0G40k.pth','pretrained_v2/f0D40k.pth':'pretraineds/hifi-gan/f0D40k.pth'}
    manifest=[]
    for remote,local in files.items():
        dest=root/'rvc/models'/local;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            tmp=dest.with_suffix(dest.suffix+'.part');print('Downloading',remote,flush=True)
            with urlopen(base+remote,timeout=90) as response,tmp.open('wb') as out:
                while block:=response.read(1024*1024):out.write(block)
            tmp.replace(dest)
        digest=hashlib.sha256(dest.read_bytes()).hexdigest();manifest.append(dict(url=base+remote,path=local,sha256=digest))
    (root/'asset-download-manifest.json').write_text(json.dumps(manifest,indent=2))
print('Backend:',root,'\nRevision:',REVISION)
