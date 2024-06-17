import os
import torch
from utils import Plotter, generate_paths, elapsed_time, plot_results, new_model
from torchmetrics.detection import mean_ap
from utils import ShipsDataset, custom_collate_fn

# !pip install pycocotools
# !pip install torchmetrics[detection]

augmentation_type = 'nothing'
# nothing
# fourier_basis_augmentation

model_name, model_root, model_filepath, log_filepath = generate_paths(augmentation_type)

# media_folder = os.path.join('media', 'model_fourier_id0')

image_mean_test = torch.tensor([0.2114, 0.2936, 0.3265])
image_std_test = torch.tensor([0.0816, 0.0745, 0.0731])

# La mean, std del test set sono all'interno del file di LOG. Trova modo per estrapolare i dati

def test(model, test_loader, device=torch.device("cpu")):
    # Normally targets should be None
    
    model._skip_resize = True
    model.eval()

    num_correct = 0
    num_examples = 0
    test_loss = 0

    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.75])
    mAP = 0    
    for i, batch in enumerate(test_loader):

        inputs = []
        targets = []
        
        for el in batch:       # el = (image,dict)
            if el[0].numel() and el[1]['labels'].numel(): # We're considering non-empty elements
                inputs.append(el[0].to(device))
                targets.append(el[1])
                
        if len(inputs) == 0:
            continue
        
        output = model(inputs)

        """
        scores come from RoIHeads class:
        pred_scores = F.softmax(class_logits, -1)
        after deleting empy boxes, low scored boxes and applying non-max suppression
        """
        
        for dic in output:
            dic["boxes"] = dic["boxes"].to(device)
            dic["labels"] = dic["labels"].to(device)
            dic["scores"] = dic["scores"].to(device)
    # # Plot Images
    # plotter = Plotter(model, threshold=0.5)
    # with torch.no_grad():
    #     for file in os.listdir('data_augmentation/imgs/src/'):
    #         # Read Image
    #         image = read_image('data_augmentation/imgs/src/'+file)
    #         # image = F.convert_image_dtype(image, dtype=torch.float)
    #         # image = transform(image)
    #         plotter(image)            
        res = metric(output,targets)
        mAP += res['map_75']
        #print(res)
        
    mAP /= len(test_loader)
    print(f"TEST, batch {i} scored {mAP:.10f}")


if __name__ == '__main__':

    checkpoint = torch.load(model_filepath, map_location=torch.device('cpu'))
    test_loader = torch.load(os.path.join(model_root, "test_loader.pt"), map_location=torch.device('cpu'))

    model = new_model()
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print Elapsed Time
    elapsed_time(log_filepath)

    # Plot Results
    plot_results(model_filepath)

    test(model, test_loader, device=torch.device('cpu'))