import unittest
import numpy as np
from voice_cover.pitch_guard import repair_source_guide
from voice_cover.register import plan_register

class PitchGuardTests(unittest.TestCase):
    def test_harmonic_jumps_and_in_between_excursions_are_repaired(self):
        melody=np.linspace(280,310,200)
        guide=melody.copy();guide[40:70]*=4;guide[100:120]*=1.4
        fixed,audit=repair_source_guide(guide,melody,melody*1.001,np.ones(200))
        np.testing.assert_array_equal(fixed,melody)
        self.assertEqual(audit['corrected_frames'],50)
    def test_supported_high_melody_is_preserved(self):
        melody=np.r_[np.full(100,290.),np.full(100,580.)]
        fixed,audit=repair_source_guide(melody,melody,melody,np.ones(200))
        np.testing.assert_array_equal(fixed,melody)
        self.assertEqual(audit['corrected_frames'],0)
    def test_disagreement_or_weak_evidence_never_forces_a_lower_note(self):
        guide=np.full(100,600.)
        for praat,strength in [(np.full(100,600.),np.ones(100)),(np.full(100,300.),np.full(100,.2))]:
            fixed,_=repair_source_guide(guide,np.full(100,300.),praat,strength)
            np.testing.assert_array_equal(fixed,guide)
    def test_silence_and_isolated_estimator_outlier_are_preserved(self):
        guide=np.full(100,300.);guide[50:70]=0
        stock=np.full(100,300.);stock[30:33]=600
        fixed,_=repair_source_guide(guide,stock,np.full(100,300.),np.ones(100))
        np.testing.assert_array_equal(fixed,guide)
    def test_short_chirp_with_sustained_independent_agreement_is_fixed(self):
        guide=np.full(100,300.);guide[30:33]=900
        fixed,audit=repair_source_guide(guide,np.full(100,300.),np.full(100,300.),np.ones(100))
        np.testing.assert_array_equal(fixed,np.full(100,300.))
        self.assertEqual(audit['corrected_frames'],3)
    def test_repaired_contour_keeps_one_register_through_the_phrase(self):
        melody=np.full(300,290.);guide=melody.copy();guide[100:180]*=4
        fixed,_=repair_source_guide(guide,melody,melody,np.ones(300))
        plan=plan_register(fixed,3)
        self.assertEqual(len(plan),1)
        self.assertTrue(plan[0]['semitones'] in [0,12])

if __name__=='__main__':unittest.main()
