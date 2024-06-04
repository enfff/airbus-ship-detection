import torch
import os
from torchvision.io import read_image
from utils import *

from Plotter import Plotter

model_filepath = os.path.join('models', 'model_fourier_id0','model.tar')
log_filepath = os.path.join('models', 'model_fourier_id0','log.txt')
# media_folder = os.path.join('media', 'model_fourier_id0')

if __name__ == '__main__':

    checkpoint = torch.load(model_filepath, map_location=torch.device('cpu'))

    model = new_model()
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print Elapsed Time
    elapsed_time(log_filepath)

    # Plot Results
    plot_results(model_filepath)

    # Plot Images
    plotter = Plotter(model)
    with torch.no_grad():
        for file in os.listdir('data_augmentation/imgs/src/'):
            # Read Image
            image = read_image('data_augmentation/imgs/src/'+file).float()/256.0
            # image = transform(image)
            plotter(image)