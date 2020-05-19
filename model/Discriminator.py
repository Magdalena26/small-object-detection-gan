import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        # self.fc1 = torch.nn.Linear(25088, 512)
        # self.lrelu1 = torch.nn.LeakyReLU(0.2, inplace=True)
        # self.fc2 = torch.nn.Linear(512, 512)
        # self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        # self.fc3 = torch.nn.Linear(512, 1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self,input_feat):
        # import ipdb
        # ipdb.set_trace()
        input_flat = input_feat.view(input_feat.size(0), -1)
        return self.main(input_flat)


