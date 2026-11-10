# MURA X-ray Ensemble

A Jupyter notebook for experimenting with binary image classification on the Stanford MURA musculoskeletal X-ray dataset. It trains EfficientNet-B4, ConvNeXt-Small, and Swin-Small with PyTorch and timm, then compares ways to combine their predictions. The workflow includes progressive image resizing for the CNNs, mixed-precision training on CUDA, checkpoint resume, and test-time augmentation.

## Set up

Use Python 3.10 or 3.11. A CUDA-capable NVIDIA GPU is recommended for the full training schedule. Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install compatible `torch` and `torchvision` builds for your system, then run the following in the repository folder.

```sh
python -m pip install -r requirements.txt
jupyter notebook xray_ensemble_v5_improved.ipynb
```

The first training run downloads pretrained weights through timm. Use the same Python environment for the notebook kernel and the packages.

## Prepare the images

Request or download MURA through the [Stanford dataset page](https://stanfordmlgroup.github.io/competitions/mura/) and follow Stanford's data terms. The notebook expects already-preprocessed images. The original experiment used contrast enhancement with CLAHE. Prepare your training and evaluation folders separately, keeping each patient's images together when creating new splits.

```text
data/
  train/
    Abnormal/
    Normal/
  test/
    Abnormal/
    Normal/
```

Place PNG, JPEG, or BMP images in the class folders or in subfolders beneath them. Change `TRAIN_DIR` and `TEST_DIR` in the configuration cell if your data lives elsewhere. Class names are case-sensitive and must match in both folders. The notebook sorts them alphabetically and uses class index 1 for its probability and binary metrics. In the example layout, class 1 is `Normal`.

## Run the experiment

Run the configuration and data-loading cells first. You should see two class names, their label mapping, and image counts for the training, validation, and evaluation splits. Use data with enough examples from both classes for the five-fold stacking comparison.

Run the training section, then the ensemble evaluation. Checkpoints are written in the working directory. The evaluation prints a strategy comparison and produces ROC and confusion-matrix plots. Dataset images, checkpoints, and generated plots are kept outside version control.

## Evaluation limits

This is an exploratory research workflow. The internal train/validation split samples individual images, so images from a study or patient can appear on both sides. Several thresholds and the highlighted ensemble are selected using evaluation labels. For an independent result, split by patient or study, fit thresholds and choose the ensemble on validation data, then evaluate once on a held-out set. The notebook is intended for research and education. Clinical use requires separate validation.

## License

The repository code is available under the [MIT License](LICENSE). MURA and the pretrained weights are provided separately under their respective terms.
