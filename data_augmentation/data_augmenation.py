import torch
import torch.fft as fft
import torchvision
import random

from torch import sin, cos

def __polar_form(complex_tensor):
    return complex_tensor.abs(), complex_tensor.angle()

def __complex_form(magnitude, angle):
    return torch.polar(magnitude,angle)

def fourier_random_noise(image):
    '''
    Scales the magnitude of each pixel of the input image's
    Fourier Transform by a random value in the range [0,1)
    '''
    # Fourier Transform
    fourier = fft.rfftn(image)
    magnitude, angle = __polar_form(fourier)

    # Apply Noise in the Frequency Domain
    noise = torch.rand(fourier.size())
    noised_magnitude = torch.mul(magnitude,noise)

    # Inverse Fourier Transform
    fourier = __complex_form(noised_magnitude,angle)
    modified_image = fft.irfftn(fourier).byte()

    return modified_image

def patch_gaussian(image, patch_size=30, sigma_max=0.2):
    '''
    Applies a Gaussian Patch of size patch_size x patch_size to the image.
    The noise of the patch can be modified by specifying its variance
    '''
    size = image.size()
    # Scale the image in range [0,1)
    min_val = 0
    max_val = 255
    image = (image-min_val)/(max_val-min_val)

    # Define Gaussian patch
    patch = torch.empty(size).normal_(0,sigma_max)
    # Sample Corner Indices
    ci = random.sample([i for i in range(size[1]-patch_size)],1)[0]
    cj = random.sample([i for i in range(size[2]-patch_size)],1)[0]
    u, v = torch.meshgrid(torch.arange(size[1]), torch.arange(size[2]),indexing='ij')
    u = torch.stack([u,u,u])
    v = torch.stack([v,v,v])
    mask = ((u<ci+patch_size)*(u>ci)*(v<cj+patch_size)*(v>cj)).int()
    patch = mask*patch

    return torch.clip(image+patch,0,1)


def auxiliary_fourier_basis_augmentation(image, l=0.3):
    '''
    Adds a Fourier Basis Function to the image
    '''
    shape = image.size()
    min_val = 0
    max_val = 255
    # Scale the image in range [0,1)
    image = (image-min_val)/(max_val-min_val)

    # Generate a frequency per channel, in the range [0, M], drawn uniformly,
    # where M is the size of the image
    f = (shape[1]-1)*torch.rand(3)
    # Generate a omega per channel, in the range [0, pi], drawn uniformly,
    w = (torch.pi-0)*torch.rand(3)

    # Sample the decay parameter from a l-exponential distribution
    sigma = torch.distributions.Exponential(1/l).sample((3,))

    # Generate basis function
    u, v = torch.meshgrid(torch.arange(shape[1]), torch.arange(shape[2]),indexing='ij')
    basis_r = sigma[0]*sin(2*torch.pi*f[0]*(u*cos(w[0])+v*sin(w[0])-torch.pi/4))
    basis_g = sigma[1]*sin(2*torch.pi*f[1]*(u*cos(w[1])+v*sin(w[1])-torch.pi/4))
    basis_b = sigma[2]*sin(2*torch.pi*f[2]*(u*cos(w[2])+v*sin(w[2])-torch.pi/4))
    noise = torch.stack([basis_r,basis_g,basis_b])

    # Modify The Image
    modified_image = image+noise

    return torch.clip(modified_image,0,1)