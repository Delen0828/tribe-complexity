"""Small real-model tests for exact early-pooling semantics; no downloaded weights."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import VJEPA2Config, VJEPA2Model
from gpu_encoder import BatchedEncoder, pooled_forward, prediction_identity, window_key
from parallelism import batches
from types import SimpleNamespace


class EncoderTests(unittest.TestCase):
    def test_pool_matches_original_hidden_states_for_batch(self):
        torch.manual_seed(42)
        config = VJEPA2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                             crop_size=32, frames_per_clip=2, patch_size=16,
                             pred_hidden_size=64, pred_num_attention_heads=4, pred_num_hidden_layers=1)
        model = VJEPA2Model(config).eval()
        pixels = torch.randn(2,2,3,32,32)
        with torch.inference_mode():
            states = model(pixel_values_videos=pixels, output_hidden_states=True,
                           skip_predictor=True).hidden_states
            expected = torch.stack(states,dim=1).mean(dim=2,keepdim=True)
            actual = pooled_forward(model,pixels)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
            # Repeated calls must remove hooks, not accumulate pooled outputs.
            torch.testing.assert_close(pooled_forward(model,pixels),expected,rtol=1e-6,atol=1e-7)

    def test_partial_batches_and_duplicate_windows(self):
        self.assertEqual(list(batches(range(5),2)), [[0,1],[2,3],[4]])
        self.assertEqual(list(batches([],2)), [])
        with self.assertRaises(ValueError):
            list(batches([1],0))
        encoder=BatchedEncoder('cpu')
        a=np.zeros((2,4,4,3),dtype=np.uint8)
        b=np.ones_like(a)
        with patch.object(encoder,'encode',return_value=torch.ones(2,3,1,4)) as encode:
            encoder.prime([a,a,b])
            self.assertEqual(len(encode.call_args.args[0]),2)
            self.assertEqual(len(encoder.cache),2)
            self.assertEqual(encoder.cache[window_key(a)].shape,(1,3,1,4))
        encoder.prime([])
        self.assertEqual(encoder.cache,{})

    def test_shared_model_and_missing_window_guard(self):
        from neuralset.extractors.video import _HFVideoModel
        from gpu_encoder import MODEL_NAME
        encoder=BatchedEncoder('cpu')
        fake_model=object()
        encoder.wrapper=SimpleNamespace(model=fake_model, processor=object())
        window=np.zeros((2,4,4,3),dtype=np.uint8)
        expected=torch.ones(1,3,1,4)
        encoder.cache[window_key(window)]=expected
        with encoder.install():
            first=_HFVideoModel(MODEL_NAME)
            second=_HFVideoModel(MODEL_NAME)
            self.assertIs(first.model,second.model)
            self.assertIs(first.predict_hidden_states(window),expected)
            with self.assertRaisesRegex(ValueError,'Unprimed'):
                second.predict_hidden_states(np.ones_like(window))

    def test_precision_identity_and_bridge_cleanup(self):
        fp32=prediction_identity('v3','sha',[255]*3)
        self.assertNotIn('precision',fp32)
        self.assertNotEqual(fp32,prediction_identity('v3','sha',[255]*3,'bf16'))
        from neuralset.extractors.video import _HFVideoModel
        original=_HFVideoModel.__init__
        encoder=BatchedEncoder('cpu')
        with self.assertRaisesRegex(ValueError,'test cleanup'):
            with encoder.install():
                self.assertIsNot(_HFVideoModel.__init__,original)
                raise ValueError('test cleanup')
        self.assertIs(_HFVideoModel.__init__,original)


if __name__ == '__main__':
    unittest.main()
