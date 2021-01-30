# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GradCam']

class GradCam():
    """
    producing 'visual explanations' for decisions made by CNN-based models
    - Refs:
        - https://github.com/ramprs/grad-cam
        + https://github.com/jacobgil/pytorch-grad-cam/
    """
    def __init__(self, device='cpu', model=None, hasrecurrent=False):
        self.device = device
        self.model = model
        self.gradients = None
        self.image = None
        self.hasrecurrent = hasrecurrent

    def __call__(self, *args, **kwargs):
        # get target feature map and model output (the score for class c before softmax y^c)

        # target feature map
        target, output = self.get_target_fmap(*args, **kwargs)

        # y^c 
        if 'index' not in kwargs.keys():
            index = torch.argmax(output)
        else:
            index = kwargs['index']
        if index == None:
            index = torch.argmax(output)
        one_hot = self.get_y_index(index=index, output=output)

        # compute the gradient w.r.t. feature map activations A^k of a convolutional layer
        one_hot.backward()

        # obtain the neuron importance weights: 
        # global-average-pool gradients over the width and height dimensions
        weights = torch.mean(self.gradients.squeeze_(0), (1, 2))       # channel, w, h

        # heatmap: weighted combination of forward activation maps, follow by a ReLU
        heatmap = torch.zeros(target.size()[1:], dtype=torch.float32).to(self.device)
        for i, w in enumerate(weights):
            heatmap += w * target[i, :, :]
        heatmap = F.relu(heatmap)        # ReLU
        return heatmap

    def get_target_fmap(self, *args, **kwargs):
        x = self.setup(*args, **kwargs)
        self._model_setup_()
        target, output = self.forward_pass(x, kwargs['target_layer'])
        target = target.squeeze(0)
        return target, output

    def get_y_index(self, index=None, output=None):
        one_hot = torch.zeros(output.size(), dtype=torch.float32)
        one_hot[index] = 1
        one_hot = torch.sum(one_hot.to(self.device) * output)
        return one_hot

    def _model_setup_(self):
        self.model.to(self.device)
        if self.hasrecurrent:
            #RuntimeError: cudnn RNN backward can only be called in training mode
            #https://github.com/pytorch/pytorch/issues/10006
            self.model.train()
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0
                elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                    module.dropout = 0
        else:
            self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.zero_grad()


    # get target feature maps and model output
    def forward_pass(self, x):
        r"""
        Note: for getting the partial derivative of class score over target feature maps,
        technically it is the same for setting requires_grad to True for either model parameters or 
        model input, since either way the target feature maps would be a intermediate node on the 
        computation graph, which is enough for attach a hook to it. 
        
        (of course only requires_grad for feature maps works as well... )

        I disabled requires_grad for all model parameters,
        only set the requires_grad for input to be true, just to be explicit 

        - General flow:

        for param in self.model.parameters():
            param.requires_grad = False
        x.requires_grad = True
        self.model.zero_grad()

        # get the feature maps we want
        >> x = get_feature(x)
        >> target_activations = x

        # register hook on the fearure maps we want the gradient for
        >> x.register_hook(self.save_gradient) 
        
        # get predicitons
        >> x = the_rest_of_the_model_before_softmax(x)
        >> torch.argsort(output, descending=True)    # top predicitons
        """
        target_activations = None
        raise NotImplementedError('Overwrite this one')
        return target_activations, x

    def save_gradient(self, grad):
        self.gradients = grad

    def setup(self, *args, **kwargs) -> torch.Tensor:
        """ 
        Prepare original image and tfed model input
        Args:
            - set self.image
            - return model input x
        * Don't forget to update hidden (cell) states for recurrent models
        """
        x.requires_grad = True
        raise NotImplementedError('Overwrite this one')
        return x

