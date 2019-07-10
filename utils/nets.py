import torch
import torch.nn as nn
from torch.utils import data

class dataset(data.Dataset):

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return self.data_y.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]


class ConvNet(nn.Module):
    def __init__(self, input_shape,  input_width, params):
        super(ConvNet, self).__init__()
        try:
            filter_size, stride_size, dilation_size, filter_size2, nb_fc_layers = params['filter_size'], params['stride'], params['dilation'], params['filter_size2'], params['nb_fc_layers']
            filter_size, stride_size, dilation_size, filter_size2, nb_fc_layers = int(filter_size), int(stride_size),  int(dilation_size),  int(filter_size2),  int(nb_fc_layers)
        except Exception as e:
            filter_size, stride_size, dilation_size, filter_size2, nb_fc_layers = 1, 1, 1, 1, 0
            
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(input_width, filter_size), stride=stride_size, padding=(0, 0), dilation=(1,dilation_size))
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=30, kernel_size=(1, filter_size2), stride=1, padding=(0, 0), dilation=(1,1))
        self.relu2 = nn.ReLU()

        self.size1 = (input_shape + 0 + 0 - dilation_size * (filter_size - 1) - 1) // stride_size + 1
        self.size2 = (self.size1 - filter_size2) // 1 + 1
        self.dp = nn.Dropout(0.2)

        self.params = params

        if nb_fc_layers == 1:
            self.out = nn.Linear(self.size2*30, 30)
            self.reluout = nn.ReLU()
            self.out2 = nn.Linear(30, 1)
        elif nb_fc_layers == 0:
            self.out = nn.Linear(self.size2*30, 1)
        
    def forward(self, x):
        
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        
        x2 = x2.view(x2.shape[0], -1)

        if self.params['nb_fc_layers'] == 1:    
            xout2 = self.out2(self.reluout(self.out(self.dp(x2))))
        else:
            xout2 = self.out(self.dp(x2))
            
        return xout2