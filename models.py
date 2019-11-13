import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab, num_classes):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        average = embedded.mean(0)
        output = self.fc1(average).squeeze(1)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)

        return output


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes, num_classes):
        super(CNN, self).__init__()

        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.from_pretrained(vocab.vectors)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim)),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(100, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        x = self.embed(x)
        x = (x.permute(1, 0, 2)).unsqueeze(1)
        x1 = self.conv1(x)
        x1, _ = torch.max(x1, 2)
        x2 = self.conv2(x)
        x2, _ = torch.max(x2, 2)
        x = torch.cat([x1, x2], 1).squeeze()
        return (self.linear(x)).squeeze()


class GRU(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, num_classes):
        super(GRU, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            F.relu()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            F.relu()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        _, h = self.rnn(x)
        h = self.fc1(h)
        h = self.fc2(h)
        return (self.fc3(h)).squeeze()


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim, num_classes):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        _, h = self.rnn(x)
        return (self.linear(h)).squeeze()
