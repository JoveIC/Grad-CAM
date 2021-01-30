# Example
import os
import cv2
import pathlib
import argparse

from PIL import Image
from torchvision import models
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize 

from gradcam import GradCam
from misc import fuse_heatmap_image

class VGGGradCam(GradCam):
    """
    Example given by demonstrating with a pretrained VGG19 model
    """
    def forward_pass(self, x, target_layer):
        x.requires_grad = True

        for name, module in self.model.named_children():
            if module == self.model.features:
                for layer, submodule in module.named_children():
                    x = submodule(x)
                    if layer == target_layer:
                        x.register_hook(self.save_gradient)
                        target_activations = x
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            elif "classifier" in name.lower():
                x = module(x)
            else:
                raise Exception('Unknown module name %s' % name)
        return target_activations, x.squeeze(0)

    def setup(self, *args, **kwargs):
        input_size = (224, 224)
        self.image = kwargs['image']
        transform = transforms.Compose([Resize(input_size), ToTensor(),
                                        Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        im = transform(self.image)
        x = im.unsqueeze(0).to(self.device)
        return x

def main():
    device = 'cuda:1'
    model = models.vgg19(pretrained=True)
    gradcam = VGGGradCam(device=device, model=model)
    img = Image.open(os.path.join(os.getcwd(), 'pics/cat_fish.jpg'))
    heatmap = gradcam(image=img, target_layer='36')
    fused = fuse_heatmap_image(img, heatmap, resize=(299, 299))
    cv2.imwrite("cat_fused.png", fused)

if __name__ == '__main__': 
    main()

