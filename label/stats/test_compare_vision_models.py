from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from compare_chart_models import Charts
from compare_vision_models import RAW_SHAPE, SIZE, cortical_layout, rasterize
from report_model_comparison import metric_table


class VisionModelTests(unittest.TestCase):
    def test_cortical_raster_keeps_each_value_in_its_own_time_channel(self):
        layout = cortical_layout()
        occupied = layout != RAW_SHAPE[1]
        self.assertEqual(int(occupied.sum()), RAW_SHAPE[1])
        values = torch.arange(1, 1 + np.prod(RAW_SHAPE), dtype=torch.float32).reshape(1, *RAW_SHAPE)
        result = rasterize(values, layout)
        self.assertEqual(tuple(result.shape), (1, 3, SIZE, SIZE))
        for time in range(RAW_SHAPE[0]):
            np.testing.assert_array_equal(result[0, time].numpy()[occupied], values[0, time].numpy()[layout[occupied]])
            self.assertTrue(torch.all(result[0, time][~occupied] == 0))

    def test_metric_table_keeps_models_in_rows_and_splits_ordered(self):
        scores = pd.DataFrame([
            ("grouped", "tribe_raw", "resnet50", "tuned", 2.4),
            ("paper", "tribe_raw", "resnet50", "tuned", 1.7),
            ("paper", "metadata", "ridge", "tuned", 4.2),
            ("paper", "tribe_parcels", "ridge", "tuned", 8.3),
        ], columns=["protocol", "features", "regressor", "setting", "mae"])
        table = metric_table(scores, "mae")
        self.assertEqual(list(table.columns), ["Metadata", "V-JEPA 2", "TRIBE v2"])
        self.assertEqual(list(table.index), ["Ridge", "ResNet-50"])
        self.assertEqual(table.at["ResNet-50", "TRIBE v2"], "1.700 / 2.400")
        self.assertEqual(table.at["ResNet-50", "Metadata"], "—")
        self.assertEqual(table.at["Ridge", "TRIBE v2"], "—")

    def test_original_image_column_pairs_direct_models_with_cortical_scores(self):
        scores = pd.DataFrame([
            ("paper", "chart_image", "vit_b_16", "tuned", 7.5),
            ("paper", "tribe_raw", "vit_b_16", "tuned", 8.5),
            ("paper", "chart_image", "vit_b_16_ridge", "tuned", 9.5),
        ], columns=["protocol", "features", "regressor", "setting", "mae"])
        table = metric_table(scores, "mae")
        self.assertEqual(list(table.columns), ["Metadata", "V-JEPA 2", "TRIBE v2", "Original chart image"])
        self.assertEqual(list(table.index), ["ViT-B/16"])
        self.assertEqual(table.at["ViT-B/16", "Original chart image"], "7.500 / —")
        self.assertEqual(table.at["ViT-B/16", "TRIBE v2"], "8.500 / —")

    def test_image_letterbox_keeps_both_ends_of_a_wide_chart(self):
        image = np.zeros((30, 224, 3), dtype=np.uint8)
        image[:, :112, 0] = 255
        image[:, 112:, 2] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chart.png"
            Image.fromarray(image).save(path)
            result = Charts([path])[0].numpy()
        np.testing.assert_array_equal(result[:, 112, 0], [255, 0, 0])
        np.testing.assert_array_equal(result[:, 112, 223], [0, 0, 255])
        np.testing.assert_array_equal(result[:, 0, 0], [255, 255, 255])


if __name__ == "__main__":
    unittest.main()
