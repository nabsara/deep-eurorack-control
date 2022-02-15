import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_eurorack_control.config import settings


#class FaderLoss(nn.Module):

#    def __init__(self):
#        super().__init__()

#    def forward(self, output, attributes):
        
#        x = output[:, :].contiguous()
#        y = attributes[:, :].max(1)[1].view(-1)

#        loss = 0
#        loss += F.cross_entropy(x, y)

#        return loss



class FaderLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, class_label, n_attr=1, n_classes=1):
        
        self.ordinal_labels = torch.ones((8, 8)).triu()

        loss = nn.BCELoss()
        b, num_attr = class_label.size()
        class_label = class_label.reshape(-1)
        label = self.ordinal_labels[class_label]
        label = label.reshape(b, num_attr, n_classes)
        label = label.to(settings.device)

        return loss(pred, label)



#class FaderDiscriminator(nn.Module):

#        def __init__(self, data_size, latent_dim=128, hidden_dim=64):
#            super(FaderDiscriminator, self).__init__()
            
#            # using similar layers as for the encoder and convolving until size 1

#            ratios = [4, 4, 4, 2]
   
#            # 1st conv layer
#            net1 = [nn.Sequential(
#                nn.Conv1d(data_size, hidden_dim, 7, stride=1, padding=3)
#            )]

#            # 4x conv batchnorm LeakyReLu
#            for i, r in enumerate(ratios):
#                net1.append(nn.Sequential(
#                    nn.Conv1d(2**i * hidden_dim, 2**(i + 1) * hidden_dim, kernel_size=2 * r + 1, stride=r, padding=r),
#                    nn.BatchNorm1d(2**(i + 1) * hidden_dim),
#                    nn.LeakyReLU(negative_slope=0.2)
#                ))

#           # conv1d batchsize x 1024 x 500 (1 band) -->  batchsize x latent_dim x 500
#           net1.append(nn.Sequential(
#               nn.Conv1d(16*hidden_dim, latent_dim, 5, stride=1, padding=2)
#           ))

#           net2 = (nn.Sequential(
#               nn.Linear(latent_dim, hidden_dim),
#               nn.LeakyReLU(0.2, inplace=True),
#                nn.Linear(hidden_dim, n_attr=1)
#            ))

#            self.net1 = nn.Sequential(*net1)
#            self.net2 = nn.Sequential(*net2)

#        def forward(self, x):
#            return self.net2(self.net1(x).view(x.size(0), hidden_dim))



class FaderDiscriminator(nn.Module):

    def __init__(self, data_size, latent_dim=128, n_layers=2, n_attr=1, n_classes=1):
        super(FaderDiscriminator, self).__init__()

        assert n_layers >= 2
        net = nn.ModuleList()
        dim_in = latent_dim 

        for _ in range(n_layers - 1):
            net.append(nn.Linear(dim_in, dim_in // 2))
            net.append(nn.Tanh())
            dim_in = dim_in // 2

        net.append(nn.Linear(dim_in, n_attr * n_classes))
        net.append(nn.Sigmoid())

        self.net = nn.Sequential(*net)

    def forward(self, z):
        outputs = []
        for layer in self.net:
            z = layer(z)
            outputs.append(z)

        return outputs        





