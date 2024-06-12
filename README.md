# airbus-ship-detection
Code for the Airbus Ship Detection on Kaggle, modified for the Machine Learning for Vision and Multimedia class at Politecnico di Torino, with a focus on Data Augmentation Techniques

- Used [schedulerplotter](https://github.com/enfff/schedulerplotter)

## TODO
- [ ] Bayesian Data Augmentation
- [x] Fourier Based Data Augmentation
- [x] Gaussian Patch Data Augmentation
- [x] Random Noise Data Augmentation
- [x] Geometric Transformations

## Targets

The target (ground-truths) are a list of dictionaries, containing three fields

``` python
"boxes": tensor([], dtype=torch.float64),
"label": tensor([], dtype=torch.int64),
"image_id": str
```

#### Previously we used git lfs
This repository used `git lfs` for tracking large files, read more about it [here](https://git-lfs.com/)

Instructions for Arch Linux
``` bash
paru -Syu git-lfs
git lfs install
```
