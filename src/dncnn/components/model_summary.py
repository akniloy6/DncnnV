"""
Generate model summary and graph of the model architechture
install torchviz & torchsummary
ak niloy~~

"""


import sys
sys.path.append('/Users/niloy/Desktop/Desktop/DncnnV/src') #define abs path


import torch
from torchviz import make_dot
from torchsummary import summary
from dncnn.components.model import DnCNN 


# Assuming the model is saved in a file named 'your_model_file.py'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)

# Provide input size (assuming input is an RGB image with dimensions 3x256x256)
input_size = (3, 256, 256)

# Use torchsummary to print the model summary
summary(model, input_size=input_size)

# After defining model
dummy_input = torch.randn(1, 3, 256, 256).to(device)
output = model(dummy_input)

# Visualize the computational graph and import as png
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render("DnCNN_model", format="svg", cleanup=True)


