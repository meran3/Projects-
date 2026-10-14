# Neural Style Transfer

A deep learning project that turns landscape photos into artwork by combining the **content** of one image with the **style** of another, using a pre-trained **VGG19** network as a feature extractor.

Course project for *Deep Learning (AI3202)*, Department of Computer Science & Artificial Intelligence, Umm Al-Qura University — 2nd semester 2026.

## How it works

1. A pre-trained VGG19 (frozen weights) extracts feature maps from the content and style images.
2. **Content loss** compares deep-layer features (`conv4_2`) of the generated image with the content image.
3. **Style loss** compares Gram matrices across `conv1_1` … `conv5_1`.
4. The generated image (initialised from the content image) is optimised directly to minimise the weighted sum of both losses.

## Experiments

Starting from a baseline (Adam, lr = 0.002, β = 5e5, 200 iterations), the notebook compares:

| Experiment | Change |
|---|---|
| TV Loss | Adds Total Variation loss for smoother output |
| Scheduler | Learning-rate scheduler |
| Weighted Style | Higher weight on early style layers (1.0 → 0.1) |
| LBFGS | LBFGS optimizer instead of Adam |
| Changed Beta / Strong Style | Style weight β = 8e5 / 1e6 |
| Content Layer | Content features from `conv5_2` |
| Multi Style | Blends features from two style images |
| Lower LR | lr = 0.001 |
| Image Size 128 | 128×128 instead of 256×256 |

### Final loss (lower is better)

| Rank | Configuration | Final loss |
|---|---|---|
| 1 | LBFGS | 24.48 |
| 2 | Multi Style | 54.53 |
| 3 | Weighted Style | 217.69 |
| 4 | Image Size 128 | 285.06 |
| 5 | Content Layer | 425.94 |
| 6 | Baseline | 444.16 |
| 7 | Scheduler | 491.66 |
| 8 | TV Loss | 529.81 |
| 9 | Lower LR | 657.90 |
| 10 | Changed Beta | 698.15 |
| 11 | Strong Style | 867.68 |

Loss values are not directly comparable across every setting (e.g. a larger β raises the loss by design), so the notebook also compares the outputs visually and through loss curves.

## Running the notebook

1. Open `DL_Project.ipynb` in Google Colab.
2. Set the runtime to GPU (*Runtime → Change runtime type → T4 GPU*).
3. Upload `style images.zip` to `/content/`.
4. Run all cells. Content images are downloaded automatically from the Kaggle dataset [`arnaud58/landscape-pictures`](https://www.kaggle.com/datasets/arnaud58/landscape-pictures).

**Stack:** Python, PyTorch, torchvision, Matplotlib, kagglehub.
