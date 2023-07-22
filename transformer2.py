import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import simple_separator
PATH = r'C:\Users\jackm\PycharmProjects\Transformer-Work\Data/'
df = pd.read_excel(PATH + 'Conversation.xlsx')
convo = list(df['question'])
with open(PATH+'convo_data.txt', 'w', encoding='utf-8') as f:
    for i in range(len(convo)):
        f.write(str(convo[i])+'\n')
f.close()

with open(PATH+'convo_data.txt', 'r', encoding='utf-8') as g:
    text = g.read()
g.close()


data = simple_separator(text)



