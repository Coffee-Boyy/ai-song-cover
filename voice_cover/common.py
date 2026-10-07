import json,os,subprocess,sys,shutil
from pathlib import Path

APPLIO_COMMIT='21392273e01da7e0169c18567d60c9b9d40282dc'
PATH_KEYS=('runtime_python','separator_python','applio','model','index','separator_models','ffmpeg')

def ensure_backend_config(root):
    """Initialize upstream's ignored runtime config without changing user settings."""
    root=Path(root);destination=root/'assets/config.json'
    if not destination.exists():
        defaults=json.loads((root/'assets/config_template.json').read_text(encoding='utf8'))
        defaults['precision']='fp32'
        write_json(destination,defaults)
    return destination

def load_profile(path):
    path=Path(path).resolve();cfg=json.loads(path.read_text(encoding='utf8'))
    for key in PATH_KEYS:
        if key not in cfg:raise ValueError(f'Missing profile key: {key}')
        raw=os.path.expandvars(os.path.expanduser(str(cfg[key])))
        if key=='ffmpeg' and shutil.which(raw):raw=shutil.which(raw)
        value=Path(raw);cfg[key]=str(value.resolve() if value.is_absolute() else (path.parent/value).resolve())
    return cfg

def reexec(cfg):
    if Path(sys.executable).resolve()!=Path(cfg['runtime_python']).resolve():
        raise SystemExit(subprocess.run([cfg['runtime_python'],'-m','voice_cover',*sys.argv[1:]]).returncode)

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
