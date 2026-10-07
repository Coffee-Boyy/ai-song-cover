"""Prepare private training data with explicit singing-only selection."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np,soundfile as sf
from .common import load_profile,write_json

def group_split(records):
    """Hold out a whole recording, never neighboring slices of a training take."""
    groups=sorted(set(r['source_sha256'] for r in records))
    heldout=groups[-1] if len(groups)>1 else None
    return [dict(r,split='validation' if r['source_sha256']==heldout else 'train') for r in records]

def validate_intervals(intervals,duration):
    intervals=sorted((float(a),float(b)) for a,b in intervals)
    for i,(a,b) in enumerate(intervals):
        if not (0<=a<b<=duration):raise ValueError('Invalid singing interval')
        if i and a<intervals[i-1][1]:raise ValueError('Overlapping singing intervals would duplicate training data')
    return intervals

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--references',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--profile',default='voice.json');p.add_argument('--isolated',action='store_true',help='References already contain isolated dry singing');p.add_argument('--singing-only',action='store_true',help='Confirm these inputs contain singing only, without speech');p.add_argument('--segments',help='JSON mapping each input basename to selected [start,end] singing intervals');a=p.parse_args()
    cfg=load_profile(a.profile);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);(out/'clips').mkdir()
    selections=json.loads(Path(a.segments).read_text()) if a.segments else None
    records=[];seen=set();recordings=[]
    for i,name in enumerate(a.references):
        source=Path(name).resolve();digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);job=out/f'reference-{i+1:02d}';job.mkdir()
        if a.isolated:
            converted=job/'input.wav'
            subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-y','-i',str(source),'-vn','-ar','44100','-ac','1','-c:a','pcm_f32le',str(converted)],check=True)
            dry=converted
        else:
            subprocess.run([cfg['separator_python'],str(Path(__file__).with_name('separate.py')),'--song',str(source),'--out',str(job/'stems'),'--profile',str(Path(a.profile).resolve())],check=True)
            subprocess.run([cfg['separator_python'],str(Path(__file__).with_name('dereverb.py')),'--vocals',str(job/'stems/vocals.wav'),'--out',str(job/'foreground'),'--profile',str(Path(a.profile).resolve())],check=True)
            dry=job/'foreground/dry-vocals.wav'
        x,sr=sf.read(dry);duration=len(x)/sr
        if not np.isfinite(x).all():raise ValueError('Nonfinite reference audio')
        intervals=validate_intervals(selections[source.name] if selections is not None else [[0,duration]],duration)
        recordings.append(dict(source=str(source),source_sha256=digest,dry=str(dry),selected_intervals=intervals))
        for start,end in intervals:
            for t in np.arange(start,end,6):
                stop=min(float(t)+6,end);clip=x[round(float(t)*sr):round(stop*sr)]
                if len(clip)<sr or np.sqrt(np.mean(clip**2))<10**(-50/20):continue
                peak=float(abs(clip).max());clipped_fraction=float(np.mean(abs(clip)>=.999))
                if clipped_fraction>.001:continue
                clip=clip*min(1,.95/max(peak,1e-9));filename=f'ref-{i+1:02d}-{len(records):04d}.wav';dest=out/'clips'/filename
                sf.write(dest,clip,sr,subtype='PCM_24')
                records.append(dict(file=str(dest),source_sha256=digest,start=float(t),end=stop,duration_seconds=len(clip)/sr,enabled=True))
    if not records:raise ValueError('No usable reference clips')
    records=group_split(records)
    manifest=dict(schema_version=1,approved=bool(a.singing_only or selections is not None),recordings=recordings,clips=records,
        singing_review='Energy/clipping filters cannot distinguish singing from ordinary speech. Review intervals or use --singing-only for already curated references.',
        validation='Recording-disjoint holdout' if len(seen)>1 else 'No independent held-out recording; supply more than one recording for validation.')
    write_json(out/'manifest.json',manifest)
    players=''.join(f'<p>{Path(r["file"]).name} · {r["split"]}</p><audio controls src="clips/{Path(r["file"]).name}"></audio>' for r in records)
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Reference review</h1><p>Check singing identity, speech, music bleed and separation damage. Edit enabled flags and approved in manifest.json before training.</p>'+players,encoding='utf8')
    print(json.dumps(dict(manifest=str(out/'manifest.json'),clips=len(records),approved=manifest['approved'])))

if __name__=='__main__':main()
