from patent_clas import algorithm, data_make
import pandas as pd
import numpy as np
import os

os.chdir("data")
df = pd.read_pickle("data.pkl")
testloader, trainloader, valloader = data_make.main(df, 0.2, 0.2)
model = algorithm.algorithm(numlabel=95, result_path = "/result")
ep = model.fit(trainloader, valloader)
model = algorithm.model(numlabel=95, model_path = f"/result/model{ep}.pth")
predict = model.predict(testloader)