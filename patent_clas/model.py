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

class Bertmodel_onetext(nn.Module):
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
        h = self.bert_model(*text)[0][:,0,:]
        h1 = self.linear1(h)
        h2 = ipc
        for layer in self.allLayer:
            h2 = layer(h2)
        h3 = torch.cat((h1,h2),1)
        output = self.linear2(self.dropout(h3))
        return output

class Bertmodel_twotexts(nn.Module):
    def __init__(self, numlabel, model_name, size, layer, dropout):
        super().__init__()
        self.bert_model1 = BertModel.from_pretrained(model_name)
        self.bert_model2 = BertModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 768)
        self.allLayer = nn.ModuleList(
            [ffnlayer(size, dropout) for _ in range(layer)],
        )
        self.linear = nn.Linear(896, numlabel)
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, text1, text2,ipc):
        h1 = self.bert_model1(**text1)[0][:,0,:]
        h2 = self.bert_model2(**text2)[0][:,0,:]
        h3 = self.linear1(h1)
        h4 = self.linear1(h2)
        h5 = ipc
        for layer in self.allLayer:
            h5 = layer(h5)
        h6 = torch.cat((h3+h4,h5),1)
        output = self.linear(self.dropout(h6))
        return output