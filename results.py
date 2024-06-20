# import os
# import torch
from utils import Plotter, generate_paths, elapsed_time, plot_results
# from torchmetrics.detection import mean_ap
# from utils import ShipsDataset, custom_collate_fn, new_model
from data_augmentation import *

augmentation_type = 'gaussian_patch'
# nothing
# fourier_basis_augmentation
# gaussian_patch

model_name, model_root, model_filepath, log_filepath = generate_paths(augmentation_type)


if __name__ == '__main__':
    # La mean, std del test set sono all'interno del file di LOG.
    # Trova modo per estrapolare i dati

    # Print Elapsed Time
    elapsed_time(log_filepath)

    # Plot Results
    plot_results(model_filepath)

    # test(model, test_loader, device=torch.device('cpu'))