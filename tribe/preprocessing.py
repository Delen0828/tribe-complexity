"""Preserve complete stimuli through V-JEPA2's 292 resize / 256 crop."""
from PIL import Image, ImageColor

PROTOCOL = 'paper_3s_t0_bt709_center256_v3'
CANVAS_SIZE = 292
STIMULUS_SIZE = 256


def prepare_stimulus(image, background='white'):
    """Fit the full image into the crop, without stretching or normalization.

    The 18-pixel outer border is discarded by the pretrained processor.
    Remaining letterbox padding uses the experiment's configurable background.
    """
    color = ImageColor.getrgb(background)
    if len(color) != 3:
        raise ValueError('Background must be an opaque RGB color')
    width, height = image.size
    scale = STIMULUS_SIZE / max(width, height)
    size = tuple(max(1, min(STIMULUS_SIZE, round(n * scale))) for n in image.size)
    # Composite transparency before resampling so transparent RGB cannot bleed.
    rgba = image.convert('RGBA')
    rgb = Image.new('RGBA', image.size, (*color, 255))
    rgb.alpha_composite(rgba)
    stimulus = rgb.convert('RGB').resize(size, Image.Resampling.LANCZOS)
    left, top = ((CANVAS_SIZE - n) // 2 for n in size)
    canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), color)
    canvas.paste(stimulus, (left, top))
    return canvas, {
        'resized_size': list(size),
        'stimulus_box': [left, top, left + size[0], top + size[1]],
        'background_rgb': list(color),
        'encoder_crop_box': [18, 18, 274, 274],
    }
