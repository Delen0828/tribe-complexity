"""Geometry and real encoder-processor regression checks (no model weights)."""
import unittest
import numpy as np
from PIL import Image
from transformers import VJEPA2VideoProcessor
from preprocessing import prepare_stimulus


class PreprocessingTests(unittest.TestCase):
    def test_full_content_survives_processor(self):
        processor = VJEPA2VideoProcessor()
        for size in [(901, 301), (301, 901), (501, 501), (1, 1000), (23, 17)]:
            with self.subTest(size=size):
                canvas, meta = prepare_stimulus(Image.new('RGB', size, 'red'))
                self.assertEqual(canvas.size, (292, 292))
                box = meta['stimulus_box']
                self.assertTrue(all(18 <= n <= 274 for n in box))
                self.assertEqual(max(meta['resized_size']), 256)
                self.assertLessEqual(abs(meta['resized_size'][0] - size[0]*256/max(size)), 1)
                self.assertLessEqual(abs(meta['resized_size'][1] - size[1]*256/max(size)), 1)
                pixels = np.asarray(canvas)
                result = processor(videos=[np.stack([pixels]*2)], return_tensors='pt')['pixel_values_videos']
                expected = pixels[18:274, 18:274].astype(np.float32)/255
                expected = (expected - np.array(processor.image_mean))/np.array(processor.image_std)
                np.testing.assert_allclose(result[0, 0].numpy(), expected.transpose(2,0,1), atol=1e-6)

    def test_transparency_and_background(self):
        canvas, meta = prepare_stimulus(Image.new('RGBA', (256,256), (255,0,0,0)), '#123456')
        self.assertEqual(meta['background_rgb'], [18,52,86])
        self.assertTrue(np.all(np.asarray(canvas) == [18,52,86]))


if __name__ == '__main__':
    unittest.main()
