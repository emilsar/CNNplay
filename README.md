# CNNplay

Browser-based, interactive playground for designing and training small CNNs on real image data. Inspired by [TensorFlow Playground](https://playground.tensorflow.org/), but with editable convolutional architectures and real images instead of toy 2D points.

Live demo: [cvmath.club/CNNplay](https://cvmath.club/CNNplay/)

## Status

**Phase 1 — Classification (current).** Edit a layer stack (Conv2D / pooling / dense / dropout / skip connections), pick hyperparameters, train MNIST in your browser via TensorFlow.js. Loss/accuracy charts and sample predictions update live.

**Phase 2 — Segmentation (next).** Same editor, urothelial cytology images with per-pixel masks. Connects to the cvmath.club textbook used by LA Mission College VOC Ed students. Architecture supports it already (ConvT, skip connections, segmentation head); the dataset loader and mask visualization are wired but the data isn't shipped yet.

## Stack

React + Vite, TensorFlow.js (WebGL backend), Recharts. No backend — pure static site.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173/CNNplay/
npm run build    # → dist/
```

See `VIABILITY.md` for the design rationale and competitive landscape.
