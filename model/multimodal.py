import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.module import Module
from model.tcn import TemporalConvNet


class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.0):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        f = self.tcn(x)
        f = f.permute(0, 2, 1) 
        return f


class MIL(nn.Module):
    def __init__(self, input_dim, h_dim=512, dropout_rate=0.0):
        super(MIL, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(input_dim, h_dim), nn.ReLU(), nn.Dropout(dropout_rate),
                                       nn.Linear(h_dim, 32), nn.Dropout(dropout_rate),
                                       nn.Linear(32, 1), nn.Sigmoid())

    def filter(self, logits, seq_len):
        instance_logits = torch.zeros(0).cuda()
        for i in range(logits.shape[0]):
            if seq_len is None:
                return logits
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        return instance_logits

    def forward(self, avf_out, seq_len):
        avf_out = self.regressor(avf_out)
        avf_out = avf_out.squeeze()
        mmil_logits = self.filter(avf_out, seq_len)
        return mmil_logits, avf_out    

class Multimodal(Module):
    def __init__(self, input_size, h_dim=32, feature_dim=64):
        super().__init__()

        self.embedding = nn.Sequential(nn.Linear(input_size, input_size//2), nn.ReLU(), nn.Dropout(0.0),
                                        nn.Linear(input_size//2, feature_dim), nn.ReLU())
        self.tcn = TCNModel(input_size=feature_dim, num_channels=[feature_dim, feature_dim, feature_dim])

        self.mil = MIL(input_dim=feature_dim, h_dim=h_dim)


    def forward(self, data, seq_len=None):

        data = self.embedding(data)
        data = self.tcn(data)

        output, avf_out = self.mil(data, seq_len)
            
        return {"output": output,
                "avf_out": avf_out,
                "satt_f": data}
    
