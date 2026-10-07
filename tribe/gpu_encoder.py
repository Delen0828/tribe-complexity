"""Persistent batched V-JEPA encoder with early mean-token pooling.

The bridge supplies pooled features to the unchanged Neuralset temporal/layer
aggregation path. Only verified static clips are primed, and cache misses fail
rather than silently encoding a different temporal window.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import numpy as np
import torch
from moviepy import VideoFileClip

MODEL_NAME = 'facebook/vjepa2-vitg-fpc64-256'
ENGINE = 'pooled_vjepa_v1'


def prediction_identity(protocol, source_sha256, background_rgb, precision='fp32'):
    identity = dict(protocol=protocol, source_sha256=source_sha256,
                    background_rgb=background_rgb, model='facebook/tribev2')
    # Old validated FP32 predictions remain reusable.
    if precision != 'fp32':
        identity['precision'] = precision
    return identity


def static_window(path, frames=64):
    """Use decoded RGB, matching Neuralset, and check every frame is identical."""
    with VideoFileClip(str(path)) as video:
        first = None
        for frame in video.iter_frames(dtype='uint8'):
            if first is None:
                first = frame.copy()
            elif not np.array_equal(frame, first):
                raise ValueError(f'Batched static encoder received a non-static clip: {path}')
    if first is None:
        raise ValueError(f'Empty clip: {path}')
    return np.repeat(first[None], frames, axis=0)


def window_key(images):
    return hashlib.sha256(images.tobytes()).hexdigest()


def pooled_forward(model, pixels):
    """Match V-JEPA hidden states: embeddings + each block output before final LN.

    Hooks retain only B x 1 x D means. Disable HF's full-state collector and
    skip the unused predictor. Never change tensors passed between blocks.
    """
    pooled = []
    handles = []
    def initial(module, inputs):
        pooled.append(inputs[0].mean(dim=1, keepdim=True))
    def output(module, inputs, result):
        pooled.append(result[0].mean(dim=1, keepdim=True))
    try:
        handles.append(model.encoder.layer[0].register_forward_pre_hook(initial))
        handles.extend(layer.register_forward_hook(output) for layer in model.encoder.layer)
        model(pixel_values_videos=pixels, output_hidden_states=False,
              output_attentions=False, skip_predictor=True)
        if len(pooled) != len(model.encoder.layer)+1:
            raise RuntimeError('Unexpected V-JEPA hidden-state layout')
        return torch.stack(pooled, dim=1).float().cpu()
    finally:
        for handle in handles:
            handle.remove()


class BatchedEncoder:
    def __init__(self, device='cuda', precision='fp32'):
        if precision not in ('fp32', 'bf16'):
            raise ValueError('Supported precisions: fp32, bf16')
        if precision == 'bf16' and (not str(device).startswith('cuda') or not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 mode requires a CUDA GPU with BF16 support')
        self.device = device
        self.precision = precision
        self.wrapper = None
        self.cache = {}
        self.encoded_windows = 0
        self.forward_batches = []
        self.model_loads = 0

    def ensure_loaded(self):
        if self.wrapper is None:
            from neuralset.extractors.video import _HFVideoModel
            # install() replaces __init__; use its saved original directly.
            wrapper = object.__new__(_HFVideoModel)
            original = getattr(self, '_original_init', _HFVideoModel.__init__)
            original(wrapper, model_name=MODEL_NAME, pretrained=True, layer_type='', num_frames=64)
            wrapper.model.to(self.device)
            wrapper.model.eval()
            self.wrapper = wrapper
            self.model_loads += 1

    def encode(self, windows):
        self.ensure_loaded()
        from neuralset.extractors.image import _fix_pixel_values
        inputs = self.wrapper.processor(videos=windows, return_tensors='pt')
        _fix_pixel_values(inputs)
        inputs = inputs.to(self.device)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.precision == 'bf16' else nullcontext()
        with torch.inference_mode(), context:
            result = pooled_forward(self.wrapper.model, inputs['pixel_values_videos'])
        if not torch.isfinite(result).all():
            raise ValueError('Non-finite encoder features')
        self.encoded_windows += len(windows)
        self.forward_batches.append(len(windows))
        return result

    def prime(self, windows):
        self.cache.clear()  # Bound CPU feature storage to the current batch.
        unique = {}
        for window in windows:
            unique.setdefault(window_key(window), window)
        if not unique:
            return
        values = self.encode(list(unique.values()))
        for key, value in zip(unique, values):
            self.cache[key] = value.unsqueeze(0)

    @contextmanager
    def install(self):
        from neuralset.extractors.video import _HFVideoModel
        original_init = _HFVideoModel.__init__
        original_predict = _HFVideoModel.predict_hidden_states
        self._original_init = original_init
        service = self
        def initialize(wrapper, model_name, pretrained=True, layer_type='', num_frames=None):
            if model_name != MODEL_NAME or not pretrained or layer_type or num_frames not in (None,64):
                raise ValueError('Batched encoder only supports the configured pretrained 64-frame V-JEPA model')
            service.ensure_loaded()
            wrapper.__dict__.update(service.wrapper.__dict__)
        def predict(wrapper, images, audio=None):
            if audio is not None:
                raise ValueError('Static batched encoder does not support audio')
            key = window_key(images)
            if key not in service.cache:
                raise ValueError('Unprimed temporal window; decoded pixels differ from the verified static clip')
            return service.cache[key]
        _HFVideoModel.__init__ = initialize
        _HFVideoModel.predict_hidden_states = predict
        try:
            yield self
        finally:
            _HFVideoModel.__init__ = original_init
            _HFVideoModel.predict_hidden_states = original_predict
            self.cache.clear()
