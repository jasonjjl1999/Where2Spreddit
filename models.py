import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab, num_classes):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc1 = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = x.mean(0)
        x = self.fc1(x).squeeze(1)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes, num_classes):
        super(CNN, self).__init__()

        self.n_filters = n_filters
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(2 * n_filters, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = (x.permute(1, 0, 2)).unsqueeze(1)
        x1 = self.conv1(x)
        x1 = torch.max(x1, 2)[0]
        x2 = self.conv2(x)
        x2 = torch.max(x2, 2)[0]
        x = torch.cat([x1, x2], 1).squeeze()
        x = x.view(-1, 2 * self.n_filters)  # For input batches of size 1, squeeze may get rid of too many dimensions

        return (self.linear(x)).squeeze()


class GRU(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, num_classes):
        super(GRU, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=2)
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        h = self.rnn(x)[1]
        return (self.linear(h)).squeeze()


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, num_classes):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=2)
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        h = self.rnn(x)[1]
        return (self.linear(h)).squeeze()


class LSTM(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, num_classes):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=2)
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        h = self.lstm(x)[1][
            0]  # lstm cell has two outputs in the form of a tuple, so we take the first element (hidden layer)
        return (self.linear(h)).squeeze()


'''
class RNNAlternate(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, num_classes):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        # self.fc2 = nn.Linear(64, 32)
        self.fc2 = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        _, h = self.rnn(x)
        h = self.fc1(h.squeeze())
        h = F.relu(h)

        return (self.fc2(h)).squeeze()

class CNN_alternate(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes, num_classes):
        super(CNN, self).__init__()

        self.n_filters = n_filters
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(2 * n_filters, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = (x.permute(1, 0, 2)).unsqueeze(1)
        x1 = self.conv1(x)
        x1, _ = torch.max(x1, 2)
        x2 = self.conv2(x)
        x2, _ = torch.max(x2, 2)
        x = torch.cat([x1, x2], 1).squeeze()
        x = x.view(-1, 2 * self.n_filters)  # For input batches of size 1, squeeze may get rid of too many dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return (self.fc3(x)).squeeze()
'''
