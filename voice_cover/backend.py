"""Scoped compatibility adapters; leave the upstream checkout unmodified."""
from contextlib import contextmanager

def exact_retrieval(pipe,feats,index,bank,index_rate):
    import torch
    key=(id(bank),str(feats.device))
    if getattr(pipe,'_cover_bank_key',None)!=key:
        pipe._cover_bank=torch.as_tensor(bank,device=feats.device,dtype=torch.float32)
        pipe._cover_bank_key=key
    reference=pipe._cover_bank;query=feats[0].float();result=[]
    for chunk in query.split(64):
        distance=chunk.square().sum(1,keepdim=True)+reference.square().sum(1)[None]-2*chunk@reference.T
        score,indices=distance.topk(min(8,len(reference)),largest=False,dim=1)
        weights=score.clamp_min(1e-5).reciprocal().square();weights/=weights.sum(1,keepdim=True)
        result.append((reference[indices]*weights[:,:,None]).sum(1))
    retrieved=torch.cat(result)[None].to(feats.dtype)
    return retrieved*index_rate+feats*(1-index_rate)

@contextmanager
def overrides(native_path,mode='exact'):
    import soundfile as sf
    from pathlib import Path
    from rvc.infer.pipeline import Pipeline
    old_write=sf.write;old_retrieval=Pipeline._retrieve_speaker_embeddings
    def write(file,*args,**kwargs):
        if Path(file).resolve()==Path(native_path).resolve():kwargs['subtype']='FLOAT'
        return old_write(file,*args,**kwargs)
    sf.write=write
    if mode=='exact':Pipeline._retrieve_speaker_embeddings=exact_retrieval
    try:yield
    finally:sf.write=old_write;Pipeline._retrieve_speaker_embeddings=old_retrieval
