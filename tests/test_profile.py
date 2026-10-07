import json,tempfile,unittest
from pathlib import Path
from voice_cover.common import load_profile,ensure_backend_config

class ProfileTests(unittest.TestCase):
    def test_backend_configuration_is_created_and_existing_settings_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'assets').mkdir()
            (root/'assets/config_template.json').write_text(json.dumps({'precision':'fp16','model_author':'None'}))
            config=ensure_backend_config(root)
            self.assertEqual(json.loads(config.read_text())['precision'],'fp32')
            config.write_text(json.dumps({'precision':'fp16','model_author':'Custom'}))
            ensure_backend_config(root)
            self.assertEqual(json.loads(config.read_text())['model_author'],'Custom')

    def test_relative_paths_are_anchored_to_profile_not_launch_directory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'voice.json';cfg={k:'relative/'+k for k in ['runtime_python','separator_python','applio','model','index','separator_models','ffmpeg']};p.write_text(json.dumps(cfg))
            loaded=load_profile(p);self.assertEqual(loaded['model'],str(Path(d)/'relative/model'))

if __name__=='__main__':unittest.main()
