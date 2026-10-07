import unittest
from voice_cover.register import plan_register,phrase_metrics,register_ramps,ramp_shift,clean_pitch_track,consensus_pitch,retry_register_allowed

class RegisterTests(unittest.TestCase):
    def test_short_safe_dip_does_not_cause_octave_excursion(self):
        p=plan_register([400.]*400+[0.]*30+[270.]*150+[0.]*30+[400.]*400,10.1)
        self.assertEqual([r['semitones'] for r in p],[0,0,0])
    def test_required_high_phrase_stays_in_safe_range(self):
        p=plan_register([400.]*400+[0.]*30+[950.]*150+[0.]*30+[400.]*400,10.1)
        self.assertEqual(p[1]['semitones'],-12)
    def test_small_pitch_errors_do_not_force_octave_retry(self):
        self.assertFalse(retry_register_allowed(dict(voiced_retention=.99,voicing_agreement=.98,median_error_cents=20)))
        self.assertTrue(retry_register_allowed(dict(voiced_retention=.5,voicing_agreement=.6,median_error_cents=20)))
    def test_false_octave_doubling_is_rejected_by_consensus(self):
        self.assertEqual(consensus_pitch([494],[988],[493]),[494])
    def test_true_high_note_is_retained(self):
        self.assertEqual(consensus_pitch([500],[1000],[1002]),[1000])
    def test_uncertain_pitch_is_not_blindly_lowered(self):
        self.assertEqual(consensus_pitch([494],[988],[float('nan')]),[988])
    def test_quiet_short_pitch_residue_is_removed(self):
        self.assertEqual(clean_pitch_track([0]*20+[600]*10+[0]*20,[.0002]*50,.0001,.001),[0.]*50)
    def test_short_clear_notes_are_retained(self):
        self.assertEqual(clean_pitch_track([440]*10,[.1]*10,.0001,.001),[440.]*10)
    def test_sustained_quiet_notes_are_retained(self):
        self.assertEqual(clean_pitch_track([440]*100,[.0002]*100,.0001,.001),[440.]*100)
    def test_smooth_transition_is_monotonic_and_has_flat_endpoints(self):
        plan=plan_register([220.]*400+[0.]*40+[470.]*400,8.4)
        ramp=register_ramps(plan)[0];a,b=ramp['start'],ramp['end']
        self.assertAlmostEqual(ramp_shift(a,ramp),12)
        self.assertAlmostEqual(ramp_shift(b,ramp),0)
        self.assertAlmostEqual(ramp_shift((a+b)/2,ramp),6)
        values=[ramp_shift(a+(b-a)*i/100,ramp) for i in range(101)]
        self.assertTrue(all(x>=y for x,y in zip(values,values[1:])))
        self.assertLess(abs(ramp_shift(a+.0001,ramp)-12),.00001)
        self.assertLess(abs(ramp_shift(b-.0001,ramp)),.00001)
        self.assertGreater(a,4);self.assertLess(b,4.4)
    def test_same_register_has_no_ramp(self):
        self.assertEqual(register_ramps(plan_register([220.]*400+[0.]*40+[220.]*400,8.4)),[])
    def test_low_verse_high_chorus(self):
        f=[220.]*400+[0.]*40+[470.]*400
        p=plan_register(f,8.4)
        self.assertEqual([r['semitones'] for r in p],[12,0])
        self.assertTrue(4<p[1]['start']<4.4)
    def test_very_high_singer_is_lowered(self):
        self.assertEqual(plan_register([950.]*500,5)[0]['semitones'],-12)
    def test_very_low_singer_can_use_two_octaves(self):
        self.assertEqual(plan_register([100.]*500,5)[0]['semitones'],24)
    def test_silence_does_not_trigger_invalid_pitch(self):
        self.assertEqual(len(plan_register([0.]*500,5)),1)
        self.assertTrue(phrase_metrics([0.]*500,[0.]*500,0,5,12)['passed'])
    def test_short_consonant_gap_does_not_split_phrase(self):
        self.assertEqual(len(plan_register([220.]*200+[0.]*10+[240.]*200,4.1)),1)
    def test_whisper_collapse_fails_even_without_clipping(self):
        self.assertFalse(phrase_metrics([440.]*500,[0.]*400+[440.]*100,0,5,0)['passed'])
    def test_wrong_octave_fails(self):
        self.assertFalse(phrase_metrics([220.]*500,[220.]*500,0,5,12)['passed'])
    def test_correct_octave_passes(self):
        self.assertTrue(phrase_metrics([220.]*500,[440.]*500,0,5,12)['passed'])
    def test_wide_phrase_requires_review(self):
        with self.assertRaises(ValueError):plan_register([80.]*250+[1400.]*250,5)

if __name__=='__main__':unittest.main()
