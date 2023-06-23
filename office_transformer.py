import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(23)
#hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-2
train_iter = 8000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("Data/office_transcripts.txt", 'r', encoding='utf-8') as f:
    text = f.read()


def preprocess_text(text):
    ind1 = 0
    ind2 = 0
    counts = 0
    newtext = ""
    while ind1 != -1:
        if newtext == "":
            ind1 = text.find("[")
            ind2 = text.find("]")
        else:
            ind1 = newtext.find("[")
            ind2 = newtext.find("]")
        if ind1 != -1:
            newtext = text[ind2+2:]
            text = text[:ind1-2]+text[ind2+2:]
            counts = counts + 1
            print(counts)
    return text

#print(text[:500])
#print("Char length of text file: {}".format(len(text)))
#print("Unique characters: ")
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
#print(len(chars))

#making encoder (string -> #) and decoder (# -> string)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#print(encode("Hello there!"))
#print(decode(encode("Hello there!")))

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
valid_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else valid_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class BiGramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #each token leads off logits for next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BiGramLanguageModel(vocab_size)

xb, yb = get_batch('train')

logits, loss = model(xb, yb)
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#simple training
for steps in range(train_iter):
    #sample batch
    xb, yb = get_batch('train')

    #evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))