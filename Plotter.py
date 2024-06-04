import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

class Plotter:
    def __init__(self, model):
        self.__model = model
        if self.__model.training:
            self.__model.eval()

    def __call__(self, img):
        prediction = self.__model([img])[0]
        boxes = prediction['boxes'].int()
        scores = prediction['scores'].tolist()

        print(f"{scores = }")
        print(f"{boxes = }")
        
        num = len(boxes)
        if num > 0:
            img = draw_bounding_boxes(
                (img*256).byte(),
                boxes, 
                labels=['{:.2f}'.format(score*100) for score in scores],
                width = 1,
                colors = 'yellow',
                font='/usr/share/fonts/cantarell/Cantarell-VF.otf', # ad enf non trova arial
                #font='arial',
                font_size = 15
            )
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        ax.imshow(img.byte().permute(1, 2, 0))
        plt.show()
        plt.close()