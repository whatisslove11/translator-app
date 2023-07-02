from model.model import device, UNK_token, EOS_token
import torch
from unidecode import unidecode
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

tokenizer = nltk.WordPunctTokenizer()
lemmatizer = nltk.WordNetLemmatizer()


def tokenize_pipeline(sentence):
    sentence = sentence.lower()
    tokens = tokenizer.tokenize(sentence)
    return [lemmatizer.lemmatize(token) for token in tokens]


def decode(src, model, word2index, index2word, max_len=128):
    model.eval()
    src = tokenize_pipeline(unidecode(src))
    src = ["<BOS>"] + src + ["<EOS>"]
    src_tensor = torch.tensor(
        [word2index[token] if token in word2index else UNK_token for token in
         src]).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor).to(device)

    with torch.no_grad():
        encoded_src = model.encoder(src_tensor, src_mask)

    trg_ids = [1]  # BOS token
    trg_tokens = []
    while len(trg_tokens) <= max_len:
        trg_tensor = torch.tensor(trg_ids).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            predictions = model.decoder(trg_tensor, encoded_src, src_mask, trg_mask)
            last_pred_id = predictions[:, -1, :].argmax(-1).item()

            if last_pred_id == EOS_token:
                break

            trg_ids.append(last_pred_id)
            trg_tokens.append(index2word[last_pred_id])

    return " ".join(trg_tokens)