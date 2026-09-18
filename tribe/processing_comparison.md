# Stimulus processing: paper, original run, and revised run

Sources: **main paper, Sections 3.2-3.3, pp. 2-3; Section 4.1, p. 3; discussion,
p. 4**; **supplement, Appendix A p. 2, Appendix B p. 3, Appendix C p. 4, Appendix D p. 5**.

| Step | Paper and supplement | Our original run | Revised run |
| --- | --- | --- | --- |
| Stimulus set | Two experiments: 60 matched Bubble/Surface pairs each; original color or grayscale stimuli | MASSVIS 751 and 4849, two different charts | Same requested MASSVIS images; not a matched experimental pair |
| Composite images | Two color Surface charts stitched side by side; single grayscale Surface chart | Each original image kept intact | Keep each MASSVIS image intact; stitching is specific to their stimuli |
| Static presentation | Constant frame for **3 seconds** (main Sec. 3.2) | Constant frame for **12 seconds** | **3 seconds** |
| Spatial preparation | Native resolution; **even-dimension crop only**, no scaling/padding | Native resolution; no even-dimension crop | Remove at most one right/bottom pixel to make dimensions even; no scaling/padding |
| Actual dimensions | Depends on source stimulus | 751: 946 x 902; 4849: 921 x 687 | 751: 946 x 902; 4849: **920 x 686** |
| Video FPS / codec | Not specified in these PDFs | 16 fps, lossless RGB H.264 | 16 fps, H.264 High/YUV420, fixed QP 10, all-intra frames, BT.709 limited range for playback compatibility; not claimed paper settings |
| Audio and task input | No audio track; omit task question; video-only model input | No audio, transcription, or separate task/question | Same |
| Model image preprocessing | Model's fixed input resizing still applies (main discussion, p. 4) | Pretrained V-JEPA2 processor | Same; native resolution refers to preparing the video, not bypassing the encoder |
| Subject / cortical output | Default unseen-subject, population-level fsaverage5; 20,484 vertices at 1 Hz | Same | Same |
| Hemodynamic delay | Already incorporated by model; no additional delay applied | Native compensation, no extra shift | Same |
| Reported response | **First prediction, t=0**; t=1 and t=2 examined separately for robustness (supp. A) | Mean of all 12 predictions | **t=0 only**; retain all 3 samples in raw NPZ files for audit, without a time-series comparison |
| Primary contrast | Surface minus Bubble at each vertex, then mean across 60 matched pairs | Difference of two temporal-mean maps, 751 minus 4849 | Difference of two **t=0 maps**, 751 minus 4849; no across-stimulus averaging |
| Global-mean sensitivity | Remove each pair's whole-brain mean (main Sec. 4.1; supp. B) | None | Additional contrast minus its vertex mean, algebraically equivalent to subtracting individually demeaned maps |
| Cortical visualization | Dorsal positive/negative contrasts separately, common 0-0.15 scale (supp. C) | Source images, lateral/medial maps, signed difference, time-series plot, metrics | Original stimulus figures plus t=0 cortical maps, signed difference, separate dorsal directions and demeaning check; common data-derived directional scale avoids clipping |
| Regional / uncertainty analysis | Destrieux parcels, 60-pair averages, BCa CIs; strongest parcels with >=50 vertices (supp. D) | Whole-cortex descriptive metrics for one pair | Not reproduced: two unrelated images cannot support their matched-pair group analysis or human-study directional agreement |

## What was adopted, and what remains unspecified

The principal changes are the three-second presentation, even-dimension cropping,
and first-timestep selection. Cropping the right and bottom edges is our explicit
choice because the paper does not specify the edge. FPS, video codec, pixel format,
and exact encoder/software revisions are not specified in either PDF; our explicitly documented encoding settings avoid pretending those details are known. The papers do
not specify the internal inference-window length, so we retain the released
checkpoint's native 100-second window, 2 Hz feature sampling, and padding behavior.

The supplement's t=0/1/2 robustness analysis is **not** a temporal average. We do not
substitute the old 12-second run's t=0: we regenerate the clips and rerun inference
because changing duration also changes the model's temporal context.

The report includes the original stimulus figures alongside neuroimaging. Raw time-indexed arrays remain available for
reproducibility, but there is no time-series plot, time-step comparison table, or
metric dashboard in the visible report. The global-mean sensitivity map is an
additional spatial check, not a statistical significance map.

This adapts their stimulus-processing protocol to MASSVIS; it does not replicate
their Bubble/Surface experiment, 60 matched pairs, or human fMRI validation.

[Main paper](../Can_a_Neural_Encoding_Model_Replicate_an_fMRI_Visu.pdf) |
[Supplement](../supplemental_materials.pdf)
