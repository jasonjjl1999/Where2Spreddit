import torchtext
from torchtext import data
import spacy
import numpy as np

from models import *

text = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
labels = data.Field(sequential=False, use_vocab=False)

train_data, val_data, test_data = data.TabularDataset.splits(
    path='./dataset', train='./training/train.csv',
    validation='./training/valid.csv', test='./training/test.csv', format='csv',
    skip_header=True, fields=[('text', text), ('label', labels)])

text.build_vocab(train_data, val_data, test_data)

text.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
vocab = text.vocab

baseline_net = torch.load('trained_models/model_baseline.pt')
cnn_net = torch.load('trained_models/model_cnn.pt')
rnn_net = torch.load('trained_models/model_rnn.pt')
gru_net = torch.load('trained_models/model_gru.pt')


def tokenizer(inp):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(inp)]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


while True:
    inp = input("Enter a sentence: ")

    tokens = tokenizer(inp)
    token_ints = [vocab.stoi[tok] for tok in tokens]
    token_tensor = torch.LongTensor(token_ints).view(-1, 1)
    lengths = torch.Tensor([len(token_ints)])
    outputs = [0, 0, 0, 0]
    outputs[0] = softmax(baseline_net(token_tensor).detach().numpy()[0])

    # TODO: See next line
    #  These should also be softmax'd. If they aren't when you test it, just add a softmax function like the above line
    outputs[1] = cnn_net(token_tensor).detach().numpy()
    outputs[2] = rnn_net(token_tensor, lengths).detach().numpy()
    outputs[3] = gru_net(token_tensor, lengths).detach().numpy()

    models = ['baseline', 'cnn', 'rnn', 'gru']
    for i in range(4):
        print('Model ' + models[i] + ': ' + '(' + str(outputs[i]) + ')')
