import torch
import pickle
from model.model import TranslationModel, device
from decode import decode


with open('de_dict.pkl', 'rb') as f:
    word2index = pickle.load(f)  # word2index

with open('en_dict.pkl', 'rb') as f:
    index2word = pickle.load(f)  # index2word

len_de = 32318
len_en = 21524

model_name = 'model.pt'

model = TranslationModel(len_de, len_en).to(device)
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
# print(model.eval())

st = 'Gutach: Noch mehr Sicherheit für Fußgänger'
print(decode(st, model, word2index, index2word))