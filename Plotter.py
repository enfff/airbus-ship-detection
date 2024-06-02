import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

class Plotter:
    def __init__(self, model):
        self.__model = model
        if self.__model.training:
            self.__model.eval()

    def __call__(self, img):
        prediction = self.__model([img])[0]
        boxes = prediction['boxes'].byte()
        scores = prediction['scores'].tolist()
        num = len(boxes)
        if num>0:
            img = draw_bounding_boxes(
                img.byte(),
                boxes, 
                labels=['{:.2f}'.format(score*100) for score in scores],
                width = 4,
                colors = 'red',
                font='Arial',
                font_size = 20
            )
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        ax.imshow(img.byte().permute(1, 2, 0))
        plt.show()
        plt.close()