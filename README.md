# Naruto Hand Gesture Recognition Using Vanilla CNN

PyTorch CNN that classifies Naruto hand signs from raw images (13 classes: bird, boar, dog, dragon, hare, horse, monkey, ox, ram, rat, snake, tiger, zero).

## Pipeline

1. **Load + augment** — reads images from `kaggle/naruto-hand-signs/data/{train,test}/<class>/`. For every image, generates 3 augmented copies (random rotate 5–10°, random scale, center crop, resize to 128×128) plus the original resized copy.
2. **Preprocess** — normalizes pixels to `[0, 1]`, converts to `(N, 3, 128, 128)` float32 tensors and int64 label tensors, holds out a 1% test split (`train_test_split`, seed 42).
3. **Model** — 5-block CNN (`Conv2d → ReLU → MaxPool2d → BatchNorm2d`, channels 64→128→256→512→1024, dropout after block 2) followed by a flatten + FC head (16384 → 1024 → 13 classes).
4. **Train** — Adam (lr=0.001), CrossEntropyLoss, batch size 32, 10 epochs, 80/20 train/val split (seed 42), Keras-style per-epoch loss/accuracy printout.
5. **Save** — weights to `model_weights.pth`.
6. **Evaluate / inspect** — grid visualization of predictions vs. true labels on the test split; standalone `predict_custom_image()` helper for running inference on an arbitrary image file.

## Requirements

```
torch torchvision opencv-python scikit-learn matplotlib pillow numpy
```
(installed via the first cell's `%pip install`)

## Expected data layout

```
kaggle/naruto-hand-signs/data/
├── train/
│   ├── bird/
│   ├── boar/
│   └── ...
└── test/
    └── ...
```

## Notes / rough edges

- Cell order assumes `classes`/`plot_sample_images` (cell 6) runs before the `X, y` referenced in cell 7 — as written, cells 4→5→8 must run before 6/7, or you'll hit `NameError`s. Recommend running top to bottom once and fixing ordering if Jupyter execution numbers look off.
- Test split is only 1% of data (`test_size=0.01`), so `x_test`/`y_test` are small — mainly useful for the visualization/sanity-check cells, not real held-out evaluation.
- `predict_custom_image()` expects a file at `testing-images/images.jpg` by default — swap the `path` variable for your own image.
- GPU is used automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU.