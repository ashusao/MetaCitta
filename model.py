#import click
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cudnn.deterministic=True

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_bn_relu, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = torch.relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, nb_filter=32, conf=(8, 2, 20, 20)):
        super(SpatialFeatureExtractor, self).__init__()
        self.nb_filter = nb_filter
        self.input_horizon = conf[0]
        self.nb_flow, self.map_height, self.map_width = conf[1], conf[2], conf[3]

        self.conv1 = conv_bn_relu(self.nb_flow, nb_filter)
        self.conv2 = conv_bn_relu(nb_filter, nb_filter)
        self.conv3 = conv_bn_relu(nb_filter, nb_filter)

    def forward(self, input_c):
        conv_op = []
        conv1_op = []
        conv2_op = []

        for i in range(self.input_horizon):
            x = input_c[:, 2 * i: 2 * i + 2, :, :].view(-1, self.nb_flow, self.map_height, self.map_width)
            out_conv1 = self.conv1(x)
            conv1_op.append(out_conv1.view(out_conv1.size(0), -1))

            out_conv2 = self.conv2(out_conv1)
            conv2_op.append(out_conv2.view(out_conv2.size(0), -1))

            out_conv3 = self.conv3(out_conv2)
            conv_op.append(out_conv3.view(out_conv3.size(0), -1))

        conv1_op = torch.stack(conv1_op, dim=0)
        #print(conv1_op.shape)
        conv1_op = conv1_op.transpose_(0, 1)

        conv2_op = torch.stack(conv2_op, dim=0)
        conv2_op = conv2_op.transpose_(0, 1)

        conv_op = torch.stack(conv_op, dim=0)
        conv_op = conv_op.transpose_(0, 1)

        return conv_op, conv1_op, conv2_op


class Body(nn.Module):
    def __init__(self, fc_size, input_size, hidden_size):
        super(Body, self).__init__()
        self.fc1 = nn.Linear(fc_size, 256, bias=True)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, conv_op):
        hidden = None
        fc1_out = F.relu(self.fc1(conv_op))
        output, hidden = self.gru(fc1_out, hidden)
        hidden = hidden.transpose_(0, 1)
        return output, hidden, fc1_out


class DomainClassifier(nn.Module):
    def __init__(self, input_size=256, conf=(8, 2, 20, 20)):
        super(DomainClassifier, self).__init__()
        self.nb_flow, self.map_height, self.map_width = conf[1], conf[2], conf[3]
        self.fc2 = nn.Linear(input_size, input_size, bias=True)
        self.fc3 = nn.Linear(input_size, self.nb_flow * self.map_height * self.map_width, bias=True)
        self.tanh = torch.tanh

    def forward(self, hidden):
        fc2_out = F.relu(self.fc2(hidden))
        out = self.fc3(fc2_out)
        out = self.tanh(out)
        out = out.view(-1, self.nb_flow, self.map_height, self.map_width)
        return out, fc2_out


class ConvLSTM(nn.Module):
    def __init__(self, nb_filter=32, conf=(8, 2, 20, 20)):
        super(ConvLSTM, self).__init__()

        self.nb_filter = nb_filter
        self.input_horizon = conf[0]
        self.nb_flow, self.map_height, self.map_width = conf[1], conf[2], conf[3]

        self.feature_extractor = SpatialFeatureExtractor(nb_filter=nb_filter, conf=conf)

        self.body = Body(self.nb_filter * self.map_height * self.map_width, input_size=256, hidden_size=256)

        self.domain_classifier = DomainClassifier(input_size=256, conf=conf)

    def forward(self, input_c):

        conv_op, conv1_op, conv2_op = self.feature_extractor(input_c)

        encoder_out, hidden, fc1_out = self.body(conv_op)

        out, fc2_out = self.domain_classifier(hidden)

        return out, conv_op, conv1_op, conv2_op, fc2_out, fc1_out

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    model = ConvLSTM(conf=(3, 2, 16, 16))
    print(model)

    #model.to(device)
    summary(model, [(6, 16, 16)], device='cpu')
    for name, param in model.named_parameters():
        print(name)