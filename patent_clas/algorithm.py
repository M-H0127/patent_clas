import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from transformers import BertJapaneseTokenizer
from patent_clas import model

class algorithm():
    def __init__(self, numlabel, twotext=False, size = 128, layer = 1, dropout = 0.2, model_name='cl-tohoku/bert-base-japanese-v2', max_length1 = 512, max_length2 = 256, result_path=None ,model_path = None, multi_gpu = False):
        if (model_path== None) & (result_path== None):
            raise Exception("result_pathかmodel_pathを入力してください") 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #GPUがつかえたらGPUを利用
        if twotext:
            self.model = model.Bertmodel_twotexts(numlabel, model_name, size, layer, dropout)
        else:
            self.model = model.Bertmodel_onetext(numlabel, model_name, size, layer, dropout)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        if model_path:
            self.model.load_state_dict(torch.load(model_path)["model_state_dict"])
        if multi_gpu > 0:
            self.model = nn.DataParallel(self.model,range(multi_gpu))
        self.multi_gpu = multi_gpu
        self.max_len1 = max_length1
        self.max_len2 = max_length2
        self.twotext = twotext
        self.loss_list = []
        self.val_list = []
        if result_path:
            self.result_dir = result_path #保存場所
            if not os.path.exists(self.result_dir): #保存場所が無かったら作成
                os.mkdir(self.result_dir)

    def fit(self, train, valid, lr=2e-5, grad_accum_step=1, early_stop_step=5):
        self.model.to(self.device)
        optimizer = optim.AdamW(params=self.model.parameters(), lr=lr) #optimizer
        criterion = nn.BCEWithLogitsLoss()
        min_score = 10000 #early_stopに使う
        early_stop = True
        ep=1
        while early_stop:
            print(f'epoch:{ep}')
            self.model.train() #モデルを訓練モードに
            running_loss = 0.0
            optimizer.zero_grad()
            j=0
            for data in tqdm(train):
                j+=1
                text1, text2, ipc, labels = data
                if self.twotext:
                    text1 = self.tokenizer(text1, max_length=self.max_len1 ,padding="max_length", truncation=True, return_tensors="pt")
                    text1 = {k: v.to(self.device) for k, v in text1.items()}
                    text2 = self.tokenizer(text2, max_length=self.max_len2 ,padding="max_length", truncation=True, return_tensors="pt")
                    text2 = {k: v.to(self.device) for k, v in text2.items()}
                    ipc = ipc.float().to(self.device)
                    outputs = self.model(text1, text2, ipc)
                else:
                    text1 = self.tokenizer(text1, max_length=self.max_len1 ,padding="max_length", truncation=True, return_tensors="pt")
                    text1 = {k: v.to(self.device) for k, v in text1.items()}
                    ipc = ipc.float().to(self.device)
                    outputs = self.model(text1, ipc)
                loss = criterion(outputs, labels.to(self.device))
                loss = loss/grad_accum_step
                loss.backward()
                if j%grad_accum_step==0:
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item()

            self.loss_list.append(running_loss/len(train))

            #検証 & early_stop
            score = self.loss_exact(valid)
            self.val_list.append(score)
            if min_score > score:
                min_score = score
                s = 0
            else:
                s+=1
            if s==early_stop_step:
                early_stop = False
            save_path = self.result_dir+f"/model{ep}.pth"
            if self.multi_gpu > 0:
                torch.save({"epoch": ep, "model_state_dict": self.model.module.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_path)
            else:
                torch.save({"epoch": ep, "model_state_dict": self.model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, save_path)
            print(running_loss/len(train))
            ep+=1
        print(f"train_loss:{self.loss_list}")
        print(f"validation_loss:{self.val_list}")
        print('Finished Training')
        print(f'Saved checkpoint at {save_path}')
        return ep-(early_stop_step+1)

    @torch.no_grad()
    def loss_exact(self, test):
        self.model.eval()
        all = 0
        vali_loss = 0
        criterion = nn.BCEWithLogitsLoss()
        for data in tqdm(test):
            text, ipc, labels = data
            if self.twotext:
                text1 = self.tokenizer(text1, max_length=self.max_len1 ,padding="max_length", truncation=True, return_tensors="pt")
                text1 = {k: v.to(self.device) for k, v in text1.items()}
                text2 = self.tokenizer(text2, max_length=self.max_len2 ,padding="max_length", truncation=True, return_tensors="pt")
                text2 = {k: v.to(self.device) for k, v in text2.items()}
                ipc = ipc.float().to(self.device)
                outputs = self.model(text1, text2, ipc)
            else:
                text1 = self.tokenizer(text1, max_length=self.max_len1 ,padding="max_length", truncation=True, return_tensors="pt")
                text1 = {k: v.to(self.device) for k, v in text1.items()}
                ipc = ipc.float().to(self.device)
                outputs = self.model(text1, ipc)
            loss = criterion(outputs, labels.to(self.device))
            all+=len(data)
            vali_loss += loss.item()
        print(f"vali_loss:{vali_loss/len(test)}")
        return vali_loss/len(test)

    @torch.no_grad()
    def predict(self, test):
        self.model.to(self.device)
        self.model.eval()
        pred_list=None
        for data in tqdm(test):
            text, ipc, labels = data
            if self.twotext:
                text1 = self.tokenizer(text1, max_length=self.max_len1 ,padding="max_length", truncation=True, return_tensors="pt")
                text1 = {k: v.to(self.device) for k, v in text1.items()}
                text2 = self.tokenizer(text2, max_length=self.max_len2 ,padding="max_length", truncation=True, return_tensors="pt")
                text2 = {k: v.to(self.device) for k, v in text2.items()}
                ipc = ipc.float().to(self.device)
                outputs = self.model(text1, text2, ipc)
            else:
                text1 = self.tokenizer(text1, max_length=self.max_len1 ,padding="max_length", truncation=True, return_tensors="pt")
                text1 = {k: v.to(self.device) for k, v in text1.items()}
                ipc = ipc.float().to(self.device)
                outputs = self.model(text1, ipc)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.to("cpu").detach().numpy()
            labels = labels.to("cpu").detach().numpy()
            labels = np.array(labels)
            if pred_list is not None:
                pred_list = np.append(pred_list, outputs,axis = 0)
                label_list = np.append(label_list, labels, axis = 0)
            else:
                pred_list=np.array(outputs)
                label_list=np.array(labels)
        return pred_list, label_list