
 # import the necessary packages
import sys
sys.path.append("..")
from pyimagesearch.nn.conv import ResNet
from keras.utils import plot_model

# initialize LeNet and then write the network architercture
# visualization graph to disk
model = ResNet.build(256, 256, 3, 205, (3, 3, 3),
                         (64, 64, 128, 256), reg=0.0005,dataset="tank")
plot_model(model, to_file="ResNet_tank.png", show_shapes=True)

