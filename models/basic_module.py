import torch
import torch.nn as nn
import time


class BasicModule(nn.Module):
    """
    encapsulate nn.Module, provide `save` and `load` function
    """
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))  # default name

    def load(self, path):
        """
        load the target model
        """
        self.load_state_dict(torch.load(path, weights_only=True))
    
    def save(self, name=None):
        """
        save the model, default name is `model_name+time`
        like `SqueezeNet_0710_23:57:29.pth`
        """
        if name is None:
            model_name = self.__class__.__name__
            prefix = "checkpoints/" + model_name + '_'
            name = time.strftime(prefix + '%m%d_%H%M%S.pth')
        torch.save(self.state_dict(),name)
        return name
    