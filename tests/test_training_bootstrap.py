import ast,unittest
from voice_cover.training_bootstrap import SingleDeviceTransport

class TrainingBootstrapTests(unittest.TestCase):
    def test_only_transport_initialization_is_guarded(self):
        tree=ast.parse('dist.init_process_group(backend="gloo")\ntrain_audio()')
        adapter=SingleDeviceTransport();fixed=adapter.visit(tree)
        self.assertEqual(adapter.changed,1);self.assertIsInstance(fixed.body[0],ast.If)
        self.assertEqual(fixed.body[1].value.func.id,'train_audio')
    def test_no_transport_means_no_unrelated_rewrites(self):
        adapter=SingleDeviceTransport();adapter.visit(ast.parse('train_audio()'))
        self.assertEqual(adapter.changed,0)
