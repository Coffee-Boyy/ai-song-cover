import unittest
import numpy as np
from voice_cover.cleanup import cleanup_audio,audit_pitch_islands,true_peak

class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.sr=16000;t=np.arange(self.sr)/self.sr;self.voice=.1*np.sin(2*np.pi*440*t)
    def test_isolated_click_is_removed(self):
        damaged=self.voice.copy();damaged[8000]+=.7
        fixed,r=cleanup_audio(damaged,self.sr,self.voice,self.sr)
        self.assertGreater(len(r['declick_repairs']),0)
        self.assertLess(abs(fixed[8000]-self.voice[8000]),.02)
        self.assertEqual(len(fixed),len(damaged))
    def test_real_source_transient_is_protected(self):
        voice=self.voice.copy();voice[8000]+=.4
        fixed,r=cleanup_audio(voice,self.sr,voice,self.sr)
        self.assertEqual(len(r['declick_repairs']),0)
    def test_clean_tone_does_not_trigger_click_repairs(self):
        fixed,r=cleanup_audio(self.voice,self.sr,self.voice,self.sr)
        self.assertEqual(len(r['declick_repairs']),0)
        self.assertEqual(len(r['level_spike_repairs']),0)
    def test_dc_is_removed(self):
        fixed,r=cleanup_audio(self.voice+.05,self.sr,self.voice,self.sr)
        self.assertTrue(r['subsonic_filter_applied'])
        self.assertLess(abs(fixed.mean()),.003)
    def test_generated_level_spike_is_attenuated(self):
        damaged=self.voice.copy();damaged[7200:8480]*=5
        fixed,r=cleanup_audio(damaged,self.sr,self.voice,self.sr)
        self.assertGreater(len(r['level_spike_repairs']),0)
        self.assertLess(np.sqrt(np.mean(fixed[7500:8100]**2)),np.sqrt(np.mean(damaged[7500:8100]**2))*.8)
    def test_real_source_level_change_is_protected(self):
        voice=self.voice.copy();voice[7200:8480]*=5
        fixed,r=cleanup_audio(voice,self.sr,voice,self.sr)
        self.assertEqual(r['level_spike_repairs'],[])
    def test_dropout_is_flagged_instead_of_inventing_a_vowel(self):
        damaged=self.voice.copy();damaged[7600:8560]=0
        fixed,r=cleanup_audio(damaged,self.sr,self.voice,self.sr)
        self.assertGreater(len(r['dropouts_requiring_review']),0)
    def test_unexplained_pitch_island_is_flagged(self):
        target=np.full(100,440.);generated=target.copy();generated[40:55]=880
        self.assertEqual(len(audit_pitch_islands(target,generated)),1)
    def test_real_melody_jump_is_protected(self):
        target=np.r_[np.full(40,440.),np.full(15,880.),np.full(45,440.)]
        self.assertEqual(audit_pitch_islands(target,target),[])
    def test_nonfinite_samples_are_rejected(self):
        with self.assertRaises(ValueError):cleanup_audio(np.full(16000,np.nan),16000)
    def test_true_peak_includes_sample_peaks(self):
        self.assertGreaterEqual(true_peak(self.voice),max(abs(self.voice))*.99)

if __name__=='__main__':unittest.main()
