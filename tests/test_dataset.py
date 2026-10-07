import unittest
from voice_cover.prepare import group_split,validate_intervals
from voice_cover.train import training_commands

class DatasetTests(unittest.TestCase):
    def test_duplicate_recording_content_stays_in_same_split(self):
        rows=group_split([dict(source_sha256='a',file='one'),dict(source_sha256='a',file='duplicate'),dict(source_sha256='b',file='other')])
        self.assertEqual(rows[0]['split'],rows[1]['split'])
        self.assertEqual(rows[2]['split'],'validation')
    def test_single_recording_has_no_fake_independent_holdout(self):
        self.assertEqual(group_split([dict(source_sha256='a')])[0]['split'],'train')
    def test_overlaps_and_out_of_bounds_are_rejected(self):
        for intervals in [[[0,4],[3,6]],[[-1,2]],[[0,11]]]:
            with self.assertRaises(ValueError):validate_intervals(intervals,10)
    def test_selected_intervals_preserve_time_order(self):
        self.assertEqual(validate_intervals([[5,9],[0,3]],10),[(0.,3.),(5.,9.)])
    def test_training_uses_only_staged_dataset_and_no_shutdown_flag(self):
        commands=training_commands(dict(applio='/backend',runtime_python='python'),'/backend/logs/test','/train-only','test',200,4,'0')
        self.assertEqual(commands[0][3],'/train-only')
        self.assertEqual(commands[1][-1],'0')
        self.assertEqual(len(commands),4)

if __name__=='__main__':unittest.main()
