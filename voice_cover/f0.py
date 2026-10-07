"""High-register tracking with independent rejection of false octave doubling."""
import numpy as np,parselmouth
from swift_f0 import SwiftF0
from rvc.lib.predictors.f0 import RMVPE
from voice_cover.register import consensus_pitch

class PitchTracker:
    def __init__(self,device):
        self.predictor=RMVPE(device,high_register={'enabled':False,'mode':'true_pitch','f0_ceil':2500})
        self.last_corrected_frames=0
        self.swift=SwiftF0()

    def get_f0(self,audio,threshold=.03):
        normal=self.predictor.get_f0(audio,threshold)
        high=self.predictor._fix_high_register(audio,normal,threshold)
        pitch=parselmouth.Sound(audio,sampling_frequency=16000).to_pitch_ac(time_step=.01,pitch_floor=65,pitch_ceiling=2500)
        independent=np.array([pitch.get_value_at_time(i*.01) for i in range(len(high))])
        self.last_stock=normal.copy()
        self.last_praat=independent.copy()
        self.last_praat_strength=np.interp(np.arange(len(high))*.01,pitch.xs(),pitch.selected_array['strength'],left=0,right=0)
        # A second neural detector supplies a vote when Praat follows a subharmonic
        # in breathy or remixed passages. Confidence and agreement are both required.
        swift=self.swift.detect(np.asarray(audio,dtype=np.float32),16000,fmin=65,fmax=2500)
        times=np.arange(len(high))*.01
        s=np.exp(np.interp(times,swift.timestamps,np.log(np.maximum(swift.pitch_hz,1))))
        confidence=np.interp(times,swift.timestamps,swift.confidence)
        agreement=(normal>0)&(confidence>=.5)
        delta=np.full(len(high),np.inf);delta[agreement]=abs(1200*np.log2(s[agreement]/normal[agreement]))
        independent=np.where(delta<=60,s,independent)
        # When stock RMVPE is unvoiced, two independent agreeing estimates may
        # reject an octave-inflated high-register fill instead of inventing a note.
        filled=(normal<=0)&(high>0)&(independent>0)&np.isfinite(independent)&(confidence>=.5)
        for i in np.where(filled)[0]:
            if abs(1200*np.log2(s[i]/independent[i]))<=60 and abs(1200*np.log2(high[i]/s[i])-1200)<=150:
                normal[i]=s[i]
        result=np.asarray(consensus_pitch(normal,high,independent))
        self.last_corrected_frames=int(np.sum(result!=high))
        return result
