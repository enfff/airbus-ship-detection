import matplotlib.pyplot as plt
import os
import torchvision

from torchvision.io import read_image
from pathlib import Path

from data_augmenation import *

for file in os.listdir('imgs/src'):
    # Read Image
    image = read_image('./imgs/src/'+file)
    shape = image.size()
    # Resize the image to be a square
    transform = torchvision.transforms.Resize(min(shape[1],shape[2]))
    image = transform(image)
    # Apply Data Augmentation
    fourier_noise = fourier_random_noise(image)
    fourier_basis = auxiliary_fourier_basis_augmentation(image)
    patch = patch_gaussian(image,300,0.2)

    # Plot Images
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(16,9)
    fig.tight_layout(pad=2)
    ax[0,0].imshow(image.permute(1, 2, 0))
    ax[0,1].imshow(fourier_noise.permute(1, 2, 0))
    ax[1,0].imshow(fourier_basis.permute(1, 2, 0))
    ax[1,1].imshow(patch.permute(1, 2, 0))
    ax[0,0].title.set_text('Original Image')
    ax[0,1].title.set_text('Fourier Random Noise')
    ax[1,0].title.set_text('Auxiliary Fourier Basis ')
    ax[1,1].title.set_text('Gaussian Patch')

    # Save Images
    file_name = Path(file).stem
    fig.savefig('imgs/target/'+file_name+'.png')

plt.show()
plt.close()