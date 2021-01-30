## Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization  

Unofficial implementation of Grad-CAM in Pytorch [<a href="https://arxiv.org/abs/1610.02391">Paper</a>]

<img src="https://github.com/JoveIC/Grad-CAM/blob/master/pics/cat_fish.jpg" width="224" height="224"> <img src="https://github.com/JoveIC/Grad-CAM/blob/master/pics/cat_heatmap.png" width="224" height="224"> <img src="https://github.com/JoveIC/Grad-CAM/blob/master/pics/cat_fused.png" width="224" height="224">

_Please check example.py to see an example_

**How to use**
```
from gradcam import GradCam

class VGGGradCam(GradCam):
    def forward_pass(self, x, target_layer):
        """
        Overwrite this one 
        Return: 
            - target_activations: activations of target feature maps
            - x: model prediction
        """
        # get the feature maps we want
        x = get_feature(x)
        target_activations = x
        # register hook on the fearure maps we want the gradient for
        x.register_hook(self.save_gradient) 
        
        # get predicitons
        x = the_rest_of_the_model_before_softmax(x)
        return target_activations, x
        
    def setup(self, *args, **kwargs):
        """
        Overwrite this one
        Return: 
            - x
        """
        # prepare the input: transform
        # prepare hidden states if have any
        x = transform(kwargs['image'])
        return x 
```

Note: For models with recurrent layers, please check [Issue](https://github.com/pytorch/pytorch/issues/10006) and modify `_model_setup_()` in gradcam.py if needed.

Ref: [jacobgil Github](https://github.com/jacobgil/pytorch-grad-cam/)
