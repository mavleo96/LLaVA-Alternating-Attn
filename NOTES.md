Training Notes:
Epochs: 1
Batch Size: 32 -> (device 2 * batch size 4 * accum steps 4)
Lora: R=32, alpha=64, dropout=0.05, bias=none

Image Aspect Ratio: nobase
Image Grid Pinpoints: (1x1),...,(6x6)
Image Merge Type: spatial_unpad

Learning Rate: 1e-5
Weight Decay: 0.0
Warmup Ratio: 0.03
LR Scheduler Type: cosine
Logging Steps: 100
Save Steps: 1000
Save Total Limit: 20

Datasets: Subset of LLaVA-OneVision-Data
Cauldron:
aokvqa: 16534 (knowledge based vqa)
chartqa: 18260 (chart based vqa)
clevr: 69995 (solid geometric objects based position vqa)
tqa/iconqa(clash): 27302 (icons)
raven: 41995 (iq questions)

Vision Flan: 186060 (vision based multiple choice questions)
