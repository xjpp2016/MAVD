import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.module import Module
from model.self_attention import Transformer

class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

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

class Unimodal(Module):
    def __init__(self, input_size, h_dim=512, feature_dim=64):
        super().__init__()

        self.embedding = Temporal(input_size,feature_dim)
        self.selfatt = Transformer(feature_dim, 2, 4, feature_dim//2, feature_dim, dropout = 0.0)

        self.mil = MIL(input_dim=feature_dim, h_dim=h_dim)

    def forward(self, data, seq_len=None, em_flag=True):

        if em_flag is True:
            data = self.embedding(data) 
            data = self.selfatt(data)

        output, avf_out = self.mil(data, seq_len)
            
        return {"output": output,
                "avf_out": avf_out,
                "satt_f": data}
