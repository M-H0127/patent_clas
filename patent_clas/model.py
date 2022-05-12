import torch
import torch.nn as nn
from transformers import BertModel


class ffnlayer(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(size, size),
        )
    def forward(self,input):
        x = input
        y =x + self.ffn(x)
        return y

class Bertmulticlassficationmodel(nn.Module):
    def __init__(self, numlabel, model_name, size, layer, dropout):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(768, 768)

        self.allLayer = nn.ModuleList(
            [ffnlayer(size, dropout) for _ in range(layer)],
        )
        self.linear2 = nn.Linear(896, numlabel)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, text, ipc):
        text, ipc = input
        h = self.bert_model(*text)[0][:,0,:]
        h1 = self.linear1(h)
        h2 = ipc
        for layer in self.allLayer:
            h2 = layer(h2)
        h3 = torch.cat((h1,h2),1)
        output = self.linear2(self.dropout(h3))
        return output

