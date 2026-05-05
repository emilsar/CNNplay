# CNNplay — Viability Assessment

**Date:** 2026-04-29
**Status:** Greenlit, phased build

## What we're building

CNNplay: a browser-based, interactive playground for designing and training CNNs on real image data. Inspired by [TensorFlow Playground](https://playground.tensorflow.org/), but instead of dense networks on toy 2D points, the user edits convolutional architectures and sees them train on real images.

## Competitive landscape

The gap is real — no existing tool combines editable architecture + live training + real image data + segmentation:

| Tool | Train in browser? | Edit architecture? | Real images? | Segmentation? |
|---|---|---|---|---|
| [TF Playground](https://playground.tensorflow.org/) | yes | yes (dense only) | no — toy 2D | no |
| [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | no (inference only) | no (fixed Tiny-VGG) | yes (10 classes) | no |
| [TensorSpace.js](https://tensorspace.org/) | no | no | yes | no |
| [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/) | yes | yes | yes (CIFAR/MNIST) | no — classification only; **unmaintained** |
| [BodyPix / DeepLab in TF.js](https://blog.tensorflow.org/2019/11/updated-bodypix-2.html) | no | no | yes | yes (pre-trained) |

## Technical viability

- **Stack:** TF.js with WebGL backend. Up to 100× faster than CPU. (WebGPU backend exists but is currently ~2/3 the speed of WebGL on small CNNs — skip for v1.)
- **Precedent:** ConvNetJS proved in-browser CNN training in 2014 with much weaker hardware. Today it's straightforward.
- **Segmentation is the hard part.** Roughly 10–100× more memory and compute per sample than classification because the output is a dense HxW mask. To make it work in-browser:
  - Tiny input sizes (64×64 or 96×96)
  - Subset of dataset (~500–1000 images)
  - Lightweight U-Net (3–4 down/up blocks, narrow channels)
  - Optionally: pre-loaded warm-start weights

## Phasing

### Phase 1 — Classification playground (MVP, in progress)
- Datasets: MNIST, Fashion-MNIST, CIFAR-10 subset, downsampled Cats vs Dogs
- User tweaks: layers (conv/pool/dense/dropout), filter count, kernel size, activation, learning rate, batch size, optimizer
- Live training loss/accuracy curve
- Per-layer feature map viz (borrow from CNN Explainer's UX)

### Phase 2 — Segmentation
- Dataset: urothelial cytology cells (binary/multi-class masks from the VocEd corpus — same images the cvmath.club textbook uses for LA Mission College VOC Ed students)
- U-Net editor: depth, channels per level, pooling vs strided conv, skip connections on/off
- Live mask overlay during training

## Recommended stack
- **TF.js (WebGL backend)** — training/inference
- **React + Vite** — UI scaffold
- **Recharts** — live training curves
- **D3** (later, for architecture graph editor in Phase 2)
- No backend — pure static site, hostable on BlueHost (cvmath.club)

## Risks
1. Segmentation training in-browser may be too slow to feel "playground-y" — set expectations or pre-train a baseline and let users fine-tune.
2. Loading 1000+ images blocks the main thread — use a Web Worker + IndexedDB cache.
3. Mobile/low-end laptops won't have GPU memory — gate the segmentation mode behind a capability check.

## Sources
- [TensorFlow Playground](https://playground.tensorflow.org/)
- [CNN Explainer (Polo Club)](https://poloclub.github.io/cnn-explainer/)
- [TensorSpace.js](https://tensorspace.org/)
- [ConvNetJS — Karpathy](https://cs.stanford.edu/people/karpathy/convnetjs/)
- [TensorFlow.js platform & backends](https://www.tensorflow.org/js/guide/platform_environment)
- [TF.js + WebGPU performance discussion](https://github.com/tensorflow/tfjs/issues/8156)
- [BodyPix browser segmentation](https://blog.tensorflow.org/2019/11/updated-bodypix-2.html)
- [Real-time semantic segmentation in browser (RefineNet)](https://github.com/hugozanini/realtime-semantic-segmentation)
- [TF Core image segmentation tutorial (U-Net reference)](https://www.tensorflow.org/tutorials/images/segmentation)
- [VocEd — applied DL for urothelial cytology segmentation](https://github.com/emilsar/VocEd)
