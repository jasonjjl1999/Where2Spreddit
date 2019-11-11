import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchtext
from torchtext import data
import spacy
import argparse
import os
from models import *
import numpy as np
import random

# Set random seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
random.seed(0)

# Set default device (for GPU usage)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torchsummary


def many_cold(one_hot):
    """
    Takes a list of one-hot vectors and turns them into numerical representations
    """
    one_hot_list = one_hot.tolist()
    labels = []
    for i in range(len(one_hot_list)):
        labels.append(one_hot_list[i].index(max(one_hot_list[i])))
    return torch.Tensor(labels)


def one_hot(x, dim):
    """
    Turns a numerical label into a one-hot vector
    """
    vec = torch.zeros(dim)
    vec[x] = 1.0
    return vec


def eval_acc(model, data, loss_fcn, model_type, type_of_eval):
    cum_corr, cum_total, cum_loss = 0, 0, 0
    for i, batch in enumerate(data, 0):
        if type_of_eval == 'val' and (i > int(0.25 * (len(data.dataset) / args.batch_size))):
            break
        elif type_of_eval == 'test' and (i > 20.0 / 64.0 * int((len(data.dataset) / args.batch_size))):
            break
        ind_batch, batch_length = batch.text
        ind_batch = ind_batch.to(device)
        label = batch.label.float()

        if model_type == 'rnn' or model_type == 'gru':
            output = model(ind_batch, batch_length)
        else:
            output = model(ind_batch)

        # Add up totals
        cum_total += label.size(0)
        cum_corr += int((many_cold(output).squeeze().float() == label).sum())
        cum_loss += loss_fcn(output, label.long().to(device))
    return float(cum_loss), float(cum_corr / cum_total)


def main(args):
    text = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    labels = data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = data.TabularDataset.splits(
        path='./dataset/training', train='train.csv',
        validation='validation.csv', test='test.csv', format='csv',
        skip_header=True, fields=[('text', text), ('label', labels)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    text.build_vocab(train_data, val_data, test_data)

    text.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = text.vocab

    print("Shape of Vocab:", text.vocab.vectors.shape)

    batch_size = args.batch_size
    lr = args.lr
    num_classes = args.num_class
    epochs = args.epochs
    model_type = args.model
    emb_dim = args.emb_dim
    rnn_hidden_dim = args.rnn_hidden_dim
    num_filt = args.num_filt
    seed = 10
    if model_type == 'cnn':
        net = CNN(emb_dim, vocab, num_filt, [2, 4], num_classes)

    elif model_type == 'rnn' or model_type == 'gru':
        net = RNN(emb_dim, vocab, rnn_hidden_dim, num_classes)
    else:
        net = Baseline(emb_dim, vocab, num_classes)

    # Use CUDA model if available:
    net.to(device)

    # Setup using Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fcn = nn.CrossEntropyLoss()

    # Plotting data
    plot_epoch = [i for i in range(1, args.epochs + 1)]
    plot_train_loss, plot_train_acc, plot_valid_loss, plot_valid_acc = [], [], [], []

    # Begin Training Loop
    for epoch in range(epochs):
        cum_loss = 0
        for (i, batch) in enumerate(train_iter, 1):
            # Setting network to training mode
            net.train()
            optimizer.zero_grad()

            # Getting data for current batch
            batch_input, batch_length = batch.text
            batch_input = batch_input.to(device)
            batch_label = nn.functional.one_hot(batch.label).float()

            # Forward step to get prediction
            if model_type == 'rnn' or model_type == 'gru':
                output = net(batch_input, batch_length)
            else:
                output = net(batch_input)

            # Loss calculation and parameter update
            loss = loss_fcn(output, many_cold(batch_label).long().to(device))
            cum_loss += loss
            loss.backward()
            optimizer.step()

        # Stats for plotting
        net.eval()
        train_loss, train_acc = eval_acc(net, train_iter, loss_fcn, model_type, 'train')
        valid_loss, valid_acc = eval_acc(net, val_iter, loss_fcn, model_type, 'val')

        plot_train_loss.append(cum_loss / (epoch + 1))
        plot_train_acc.append(train_acc)
        plot_valid_loss.append(valid_loss / (epoch + 1))
        plot_valid_acc.append(valid_acc)

        # Print progress per batch to monitor progress
        print('[%d] Train Loss: %.3f  Valid Loss: %.3f Train Acc: %.3f Valid Acc: %3f ' % (epoch + 1,
                                                                                           cum_loss / (epoch + 1),
                                                                                           valid_loss / (epoch + 1),
                                                                                           train_acc, valid_acc))
    # Final Test Accuracy
    test_loss, test_acc = eval_acc(net, test_iter, loss_fcn, model_type, 'test')
    print('Final Test Loss: ' + str(test_loss / (epoch + 1)) + ', Final Test Acc: ' + str(test_acc))

    # Saving model
    # torch.save(net, 'model_rnn.pt')

    # Plot Losses and Accuracy
    plt.figure()
    plt.plot(plot_epoch, plot_train_loss, label='Training Loss')
    plt.plot(plot_epoch, plot_valid_loss, label='Validation Loss')
    plt.title("Losses as Function of Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(plot_epoch, plot_train_acc, label='Training Accuracy')
    plt.plot(plot_epoch, plot_valid_acc, label='Validation Accuracy')
    plt.title("Accuracy as Function of Epoch")
    plt.ylim(0, 1.01)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    print(net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)
    parser.add_argument('--num-class', type=int, default=10)

    args = parser.parse_args()

    main(args)

'''
--model baseline --lr 0.01 --epochs 100
--model cnn --lr 0.01 --epochs 100

'''
