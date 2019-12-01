import torchtext
from torchtext import data
import spacy

from main_data_collection import subreddits  # Import this list to get the actual (subreddit) names of labels
from models import *
from filter import *

from filter import *

# Set default device (for GPU usage)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torchsummary

text = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
labels = data.Field(sequential=False, use_vocab=False)

train_data, val_data, test_data = data.TabularDataset.splits(
    path='./dataset', train='./training/train.csv',
    validation='./training/valid.csv', test='./training/test.csv', format='csv',
    skip_header=True, fields=[('text', text), ('label', labels)])

text.build_vocab(train_data, val_data, test_data)

text.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
vocab = text.vocab

baseline_net = torch.load('trained_models/model_baseline.pt', map_location=torch.device(device))
cnn_net = torch.load('trained_models/model_cnn.pt', map_location=torch.device(device))
rnn_net = torch.load('trained_models/model_rnn.pt', map_location=torch.device(device))
gru_net = torch.load('trained_models/model_gru.pt', map_location=torch.device(device))
lstm_net = torch.load('trained_models/model_lstm.pt', map_location=torch.device(device))


def tokenizer(inp):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(inp)]


while True:
    inp = input("Enter a sentence: ")
    # Why do airplanes need to fly so high
    inp = filter(inp)

    tokens = tokenizer(inp)
    token_ints = [vocab.stoi[tok] for tok in tokens]
    token_tensor = torch.LongTensor(token_ints).view(-1, 1)
    token_tensor = token_tensor.to(device)
    lengths = torch.Tensor([len(token_ints)])
    outputs = [0, 0, 0, 0, 0]
    outputs[0] = baseline_net(token_tensor).cpu().detach().numpy()[0]
    outputs[1] = cnn_net(token_tensor).cpu().detach().numpy()
    outputs[2] = rnn_net(token_tensor, lengths).cpu().detach().numpy()
    outputs[3] = gru_net(token_tensor, lengths).cpu().detach().numpy()
    outputs[4] = lstm_net(token_tensor, lengths).cpu().detach().numpy()

    models = ['baseline', 'cnn', 'rnn', 'gru', 'lstm']

    for i in range(len(outputs)):
        outputs[i] = [100 * prediction for prediction in outputs[i]]  # Multiply every value by 100 to get a percentage
        # print('Model ' + models[i] + ': ' + '(' + str(outputs[i]) + ')')

    for i in range(len(outputs)):  # Show the top 3 predictions
        for j in range(len(outputs[i])):
            outputs[i][j] = (outputs[i][j], j)  # Add index of element to tuple

        outputs[i].sort(key=lambda x: x[0], reverse=True)  # Sort by first element
        outputs[i] = outputs[i][0:3]  # Get the top 3 tuples

    print()
    for i in range(len(outputs)):
        print(models[i] + ' prediction:')
        for top in range(3):
            print(subreddits[outputs[i][top][1]] + ' with probability ' + str(round(outputs[i][top][0], 3)) + '%')
        print()

    print()
