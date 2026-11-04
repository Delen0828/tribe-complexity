"""SAM backends returning compact masks in native full-figure coordinates."""
from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import numpy as np
from common import CV, encode_mask, file_hash

os.environ.setdefault("HF_HOME", str(CV / "cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(CV / "cache/torch"))
os.environ.setdefault("TRITON_CACHE_DIR", str(CV / "cache/triton"))
os.environ.setdefault("CUDA_CACHE_PATH", str(CV / "cache/cuda"))


class SegmentationBackend(ABC):
    @abstractmethod
    def segment(self, image, panel):
        """Yield candidate dicts with native-coordinate RLE, score and provenance."""

    @abstractmethod
    def provenance(self):
        pass


def packages():
    result = {}
    for package in ["torch", "torchvision", "numpy", "pillow", "SAM-2", "sam3"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def candidate(mask, score, origin, source, **extra):
    mask = np.asarray(mask, dtype=bool)
    rle = encode_mask(mask)
    rle["origin"][0] += origin[0]
    rle["origin"][1] += origin[1]
    return {"rle": rle, "area": int(mask.sum()), "score": float(score), "origin": source, **extra}


class SAM2Backend(SegmentationBackend):
    def __init__(self, config, checkpoint=None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.torch = torch
        self.checkpoint = Path(checkpoint or CV / "checkpoints/sam2.1_hiera_small.pt")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process. Run GPU inference with host device access.")
        self.model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", str(self.checkpoint), device="cuda", apply_postprocessing=False)
        self.generator = SAM2AutomaticMaskGenerator(self.model, **config["sam2"])
        self.precision = config.get("precision", "bf16")
        self.config = config
        self.memory_retries = []

    def provenance(self):
        import torch
        revision = (CV / "upstream-sam2-revision.txt").read_text().strip()
        return {"backend": "sam2.1-hiera-small", "checkpoint_sha256": file_hash(self.checkpoint), "upstream_revision": revision, "packages": packages(), "gpu": torch.cuda.get_device_name(), "internal_image_size": self.model.image_size, "precision": self.precision, "semantic_prompt_support": False, "exemplars": "geometric box seeds and model-assisted visual triage", "cuda_extension": "optional connected-component postprocessing disabled"}

    def segment(self, image, panel):
        from sam2.utils.amg import rle_to_mask
        x0, y0, x1, y1 = map(int, panel["bbox"])
        crop = np.array(image.crop((x0, y0, x1, y1)).convert("RGB"))
        ctx = self.torch.autocast("cuda", dtype=self.torch.bfloat16) if self.precision == "bf16" else contextlib.nullcontext()
        with self.torch.inference_mode(), ctx:
            while True:
                try:
                    raw = self.generator.generate(crop)
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or self.generator.points_per_batch <= 1:
                        raise
                    self.generator.points_per_batch = max(1,self.generator.points_per_batch//2)
                    self.memory_retries.append({"panel_id":panel["panel_id"],"retry_points_per_batch":self.generator.points_per_batch})
                    self.generator.predictor.reset_predictor()
                    self.torch.cuda.empty_cache()
            for item in raw:
                yield candidate(rle_to_mask(item["segmentation"]), item["predicted_iou"], [x0, y0], "automatic_grid", stability_score=float(item["stability_score"]), points_per_batch=self.generator.points_per_batch)
            del raw
            # AMG resets its predictor. Re-encode once for all exemplar-box seeds.
            families = panel.get("mark_families", [])
            if families:
                predictor = self.generator.predictor
                predictor.set_image(crop)
                for family in families:
                    for box in family["exemplar_boxes"]:
                        local_box = np.array([box[0]-x0, box[1]-y0, box[2]-x0, box[3]-y0], dtype=np.float32)
                        masks, scores, _ = predictor.predict(box=local_box, multimask_output=False)
                        yield candidate(masks[0], scores[0], [x0, y0], "exemplar_box", exemplar_family=family["name"])
                predictor.reset_predictor()


class SAM3Backend(SegmentationBackend):
    def __init__(self, config, checkpoint=None):
        import torch
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise RuntimeError("Install the official facebookresearch/sam3 package in this environment before switching backends; see README.md") from e
        self.checkpoint = Path(checkpoint or CV / "checkpoints/sam3.pt")
        if not self.checkpoint.is_file():
            raise RuntimeError("SAM 3 requires approved checkpoint access. Supply --checkpoint after obtaining the weights.")
        self.torch = torch
        self.model = build_sam3_image_model(checkpoint_path=str(self.checkpoint), load_from_HF=False, device="cuda")
        class CompactProcessor(Sam3Processor):
            # Avoid a [instances, native height, native width] float tensor on GPU.
            # Equivalent to the official postprocessing, with native interpolation
            # performed one mask at a time on CPU after concept prompting.
            def _forward_grounding(processor, state):
                from sam3.model import box_ops
                outputs = processor.model.forward_grounding(backbone_out=state["backbone_out"], find_input=processor.find_stage, geometric_prompt=state["geometric_prompt"], find_target=None)
                probabilities = (outputs["pred_logits"].sigmoid() * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
                keep = probabilities > processor.confidence_threshold
                state["low_res_masks"] = outputs["pred_masks"][keep].detach().cpu()
                state["scores"] = probabilities[keep].detach().cpu()
                boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][keep])
                scale = torch.tensor([state["original_width"], state["original_height"]]*2, device=boxes.device)
                state["boxes"] = (boxes*scale).detach().cpu()
                return state
        self.processor = CompactProcessor(self.model)
        self.config = config

    def provenance(self):
        import torch
        return {"backend": "sam3", "checkpoint_sha256": file_hash(self.checkpoint), "packages": packages(), "gpu": torch.cuda.get_device_name(), "internal_image_size": 1008, "semantic_prompt_support": True, "precision": "fp32"}

    def segment(self, image, panel):
        x0, y0, x1, y1 = map(int, panel["bbox"])
        crop = image.crop((x0, y0, x1, y1))
        state = self.processor.set_image(crop)
        for family in panel["mark_families"]:
            self.processor.reset_all_prompts(state)
            self.processor.set_text_prompt(family["prompt"], state)
            for b in family["exemplar_boxes"]:
                cx, cy = ((b[0]+b[2])/2-x0)/crop.width, ((b[1]+b[3])/2-y0)/crop.height
                self.processor.add_geometric_prompt([cx, cy, (b[2]-b[0])/crop.width, (b[3]-b[1])/crop.height], True, state)
            for logits, score in zip(state["low_res_masks"], state["scores"]):
                native = self.torch.nn.functional.interpolate(logits.reshape(1,1,*logits.shape[-2:]).float(), (crop.height,crop.width), mode="bilinear", align_corners=False)[0,0] > 0
                yield candidate(native.numpy(), score.item(), [x0, y0], "text_plus_exemplars", exemplar_family=family["name"])


def create_backend(name, config, checkpoint=None):
    if name == "sam2.1-hiera-small":
        return SAM2Backend(config, checkpoint)
    if name == "sam3":
        return SAM3Backend(config, checkpoint)
    raise ValueError(f"Unknown backend {name}")
