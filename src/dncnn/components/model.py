import sys

sys.path.append('/Users/niloy/Desktop/Desktop/DncnnV/src')


import torch.nn as nn
from dncnn.utils.common import read_config
from dncnn.utils.logger import logger


config = read_config("../../../config/config.yaml")
model_config = config["model_config"]



"""
Model summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 256]           1,728
              ReLU-2         [-1, 64, 256, 256]               0
            Conv2d-3         [-1, 64, 256, 256]          36,864
       BatchNorm2d-4         [-1, 64, 256, 256]             128
              ReLU-5         [-1, 64, 256, 256]               0
            Conv2d-6         [-1, 64, 256, 256]          36,864
       BatchNorm2d-7         [-1, 64, 256, 256]             128
              ReLU-8         [-1, 64, 256, 256]               0
            Conv2d-9         [-1, 64, 256, 256]          36,864
      BatchNorm2d-10         [-1, 64, 256, 256]             128
             ReLU-11         [-1, 64, 256, 256]               0
           Conv2d-12         [-1, 64, 256, 256]          36,864
      BatchNorm2d-13         [-1, 64, 256, 256]             128
             ReLU-14         [-1, 64, 256, 256]               0
           Conv2d-15         [-1, 64, 256, 256]          36,864
      BatchNorm2d-16         [-1, 64, 256, 256]             128
             ReLU-17         [-1, 64, 256, 256]               0
           Conv2d-18         [-1, 64, 256, 256]          36,864
      BatchNorm2d-19         [-1, 64, 256, 256]             128
             ReLU-20         [-1, 64, 256, 256]               0
           Conv2d-21         [-1, 64, 256, 256]          36,864
      BatchNorm2d-22         [-1, 64, 256, 256]             128
             ReLU-23         [-1, 64, 256, 256]               0
           Conv2d-24         [-1, 64, 256, 256]          36,864
      BatchNorm2d-25         [-1, 64, 256, 256]             128
             ReLU-26         [-1, 64, 256, 256]               0
           Conv2d-27         [-1, 64, 256, 256]          36,864
      BatchNorm2d-28         [-1, 64, 256, 256]             128
             ReLU-29         [-1, 64, 256, 256]               0
           Conv2d-30         [-1, 64, 256, 256]          36,864
      BatchNorm2d-31         [-1, 64, 256, 256]             128
             ReLU-32         [-1, 64, 256, 256]               0
           Conv2d-33         [-1, 64, 256, 256]          36,864
      BatchNorm2d-34         [-1, 64, 256, 256]             128
             ReLU-35         [-1, 64, 256, 256]               0
           Conv2d-36         [-1, 64, 256, 256]          36,864
      BatchNorm2d-37         [-1, 64, 256, 256]             128
             ReLU-38         [-1, 64, 256, 256]               0
           Conv2d-39         [-1, 64, 256, 256]          36,864
      BatchNorm2d-40         [-1, 64, 256, 256]             128
             ReLU-41         [-1, 64, 256, 256]               0
           Conv2d-42         [-1, 64, 256, 256]          36,864
      BatchNorm2d-43         [-1, 64, 256, 256]             128
             ReLU-44         [-1, 64, 256, 256]               0
           Conv2d-45         [-1, 64, 256, 256]          36,864
      BatchNorm2d-46         [-1, 64, 256, 256]             128
             ReLU-47         [-1, 64, 256, 256]               0
           Conv2d-48          [-1, 3, 256, 256]           1,728
  ConvTranspose2d-49          [-1, 3, 512, 512]              84
================================================================
Total params: 558,420
Trainable params: 558,420
Non-trainable params: 0





"""


class DnCNN(nn.Module):
    """
    A class used to implement the DnCNN model for super resolution tasks.

    ...

    Attributes
    ----------
    channels : int
        the number of channels in the convolutional layers (default is 64)
    num_of_layers : int
        the number of layers in the model (default is 17)

    Methods
    -------
    forward(x)
        Defines the computation performed at every call.
    """

    def __init__(
        self,
        channels=model_config["start_channels"],
        num_of_layers=model_config["depth"],
        up_scale=model_config["up_scale"],
        mood=model_config["mood"],
        weight_initilization=model_config["weight_initilization"],
    ):
        """
        Constructs all the necessary attributes for the DnCNN object.

        Parameters
        ----------
            channels : int, optional
                the number of channels in the convolutional layers (default is 64)
            num_of_layers : int, optional
                the number of layers in the model (default is 17)
        """
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=3,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        # for m in layers:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        
       

        if weight_initilization : 
            logger.info("Weight initilization is on")
            for i in range(up_scale):
                conv_transpose = nn.ConvTranspose2d(
                    3, 3, 3, stride=2, padding=1, output_padding=1
                )
                layers.append(conv_transpose)


        if mood =='train':
            for m in layers:
                logger.info("Weight initilization is on for layers")
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                elif isinstance(m, nn.Conv2d):
                    logger.info("Weight initilization is on for conv2d layers")
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                elif isinstance(m, nn.BatchNorm2d):
                    logger.info("Weight initilization is on for batchnorm layers")
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            
                
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
            x : torch.Tensor
                the input tensor

        Returns
        -------
        out : torch.Tensor
            the output of the model
        """
        out = self.dncnn(x)
        return out




class Unet: 
    def __init__(self) -> None:
        pass

    def forward(self, x): 
        pass


class Resnet: 
    def __init__(self) -> None:
        pass

    def forward(self, x): 
        pass

class Restored: 
    def __init__(self) -> None:
        pass

    def forward(self, x): 
        pass 