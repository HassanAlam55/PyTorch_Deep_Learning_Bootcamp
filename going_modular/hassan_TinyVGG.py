# # do the CNN to classify into one of e classes
#  with model stacking to improve performance
import torch
from torch import nn
class HassanFood(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """    
    def __init__ (self, input_shape, hidden_units, output_shape,
                  in_ConvNN_ker_size: int = 3,
                  in_ConvNN_stirde:int = 1,
                  in_ConvNN_pad: int = 1,
                  in_MAXP_KerSize: int = 4,
                  in_MAXP_stride = 1,
                  in_batch_size = 32):
        super().__init__()
        self.conv1= nn.Conv2d(
                in_channels = input_shape,
                out_channels = hidden_units,
                kernel_size = in_ConvNN_ker_size,
                stride = in_ConvNN_stirde,
                padding = in_ConvNN_pad 
                )
        self.conv2 = nn.Conv2d(
            in_channels = hidden_units,
            out_channels=hidden_units,
            kernel_size=in_ConvNN_ker_size,
            stride=in_ConvNN_stirde,
            padding=in_ConvNN_pad
        )
        self.relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size=in_MAXP_KerSize,
                                    stride = in_MAXP_stride)
        self.lazydense = nn.LazyLinear(out_features=output_shape)

    def forward (self, x):
        x = self.lazydense(self.flatten(self.maxpool(self.relu(self.conv2(self.maxpool(self.relu(self.conv1(x))))))))

        return (x)
    
