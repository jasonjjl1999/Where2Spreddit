import random

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch.optim as optim
import torchtext
from torchtext import legacy

from redditscore.tokenizer import CrazyTokenizer

from models import *
from confusion import plot_confusion_matrix
from main_data_collection import subreddits  # Import this list to get the actual (subreddit) names of labels

from nltk import sent_tokenize

# Set random seeds
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)

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
        ind_batch, batch_length = batch.text
        ind_batch = ind_batch.to(device)
        label = batch.label.float()

        if model_type == 'rnn' or model_type == 'gru' or model_type == 'lstm':
            output = model(ind_batch, batch_length)
        else:
            output = model(ind_batch)

        # Add up totals
        cum_total += label.size(0)
        cum_corr += int((many_cold(output).squeeze().float() == label).sum())
        cum_loss += loss_fcn(output, label.long().to(device))
    return float(cum_loss), float(cum_corr / cum_total)


def main(args):
    train_data_count = pd.read_csv('dataset/training/train.csv')
    val_data_count = pd.read_csv('dataset/training/valid.csv')
    test_data_count = pd.read_csv('dataset/training/test.csv')

    print('The count for each label in the training set is:')
    print(train_data_count['label'].value_counts())
    print()
    print('The count for each label in the validation set is:')
    print(val_data_count['label'].value_counts())
    print()
    print('The count for each label in the testing set is:')
    print(test_data_count['label'].value_counts())
    print()

    if args.tokenizer == 'crazy':
        print('The tokenizer is: CrazyTokenizer \n')
        tokenizer = CrazyTokenizer().tokenize
    if args.tokenizer == 'nltk':
        print('The tokenizer is: NLTK \n')
        tokenizer = sent_tokenize
    else:
        print('The tokenizer is: spacy \n')
        tokenizer = 'spacy'

    print('The model used is:', args.model, '\n')

    text = legacy.data.Field(sequential=True, lower=True, include_lengths=True)
    labels = legacy.data.Field(sequential=False, use_vocab=False)

    train_data, val_data, test_data = legacy.data.TabularDataset.splits(
        path='./dataset', train='./training/train.csv',
        validation='./training/valid.csv', test='./training/test.csv', format='csv',
        skip_header=True, fields=[('text', text), ('label', labels)])

    train_iter, val_iter, test_iter = legacy.data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    text.build_vocab(train_data, val_data, test_data)

    text.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = text.vocab

    print('Shape of Vocab:', text.vocab.vectors.shape, '\n')

    lr = args.lr
    num_classes = args.num_class
    epochs = args.epochs
    model_type = args.model
    emb_dim = args.emb_dim
    rnn_hidden_dim = args.rnn_hidden_dim
    num_filt = args.num_filt

    if model_type == 'cnn':
        net = CNN(emb_dim, vocab, num_filt, [3, 4], num_classes)
    elif model_type == 'rnn':
        net = RNN(emb_dim, vocab, rnn_hidden_dim, num_classes)
    elif model_type == 'gru':
        net = GRU(emb_dim, vocab, rnn_hidden_dim, num_classes)
    elif model_type == 'lstm':
        net = LSTM(emb_dim, vocab, rnn_hidden_dim, num_classes)
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

    print('---------- TRAINING LOOP ---------- \n')

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
            if model_type == 'rnn' or model_type == 'gru' or model_type == 'lstm':
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

        plot_train_loss.append(train_loss / (train_data_count.shape[0]))
        plot_train_acc.append(train_acc)
        plot_valid_loss.append(valid_loss / (val_data_count.shape[0]))
        plot_valid_acc.append(valid_acc)

        # Print progress per batch to monitor progress
        print('[%d] Train Loss: %.3f  Valid Loss: %.3f Train Acc: %.3f Valid Acc: %3f ' % (epoch + 1,
                                                                                           cum_loss / (epoch + 1),
                                                                                           valid_loss / (epoch + 1),
                                                                                           train_acc, valid_acc))
    # Final Results
    test_loss, test_acc = eval_acc(net, test_iter, loss_fcn, model_type, 'test')
    val_loss, val_acc = eval_acc(net, val_iter, loss_fcn, model_type, 'valid')
    train_loss, train_acc = eval_acc(net, train_iter, loss_fcn, model_type, 'train')

    print()
    print('---------- FINAL RESULTS ----------')
    print()
    print('Final Training Loss: ' + str(train_loss / (epoch + 1)) + ', Final Training Acc: ' + str(train_acc))
    print('Final Validation Loss: ' + str(val_loss / (epoch + 1)) + ', Final Validation Acc: ' + str(val_acc))
    print('Final Test Loss: ' + str(test_loss / (epoch + 1)) + ', Final Test Acc: ' + str(test_acc))

    '''

    train_iter, val_iter, test_iter = legacy.data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(len(train_data), len(val_data), len(test_data)),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)


    for (i, batch) in enumerate(train_iter, 1):
        # Setting network to eval mode
        net.eval()

        # Getting data for current batch
        batch_input, batch_length = batch.text
        batch_input = batch_input.to(device)
        batch_label = nn.functional.one_hot(batch.label).float()

        # Forward step to get prediction
        if model_type == 'rnn' or model_type == 'gru':
            output = net(batch_input, batch_length)
        else:
            output = net(batch_input)

    outputs = many_cold(output)
    batch_label = many_cold(batch_label)
    print("Below is Confusion Matrix for Training Set")
    print(confusion_matrix(batch_label, outputs))
    '''

    batch_label = torch.empty(0).to(device).float()
    output = torch.empty(0).to(device)

    for (i, batch) in enumerate(val_iter, 1):

        # Getting data for current batch
        batch_input, batch_length = batch.text
        batch_input = batch_input.to(device)
        batch_label = torch.cat((batch_label, batch.label.to(device).float()))

        # Forward step to get prediction
        if model_type == 'rnn' or model_type == 'gru' or model_type == 'lstm':
            output = torch.cat((output, net(batch_input, batch_length)))
        else:
            output = torch.cat((output, net(batch_input)))

    outputs = many_cold(output)

    # Print number of trainable parameters in the model
    print()
    print('The number of trainable parameters in the model is:')
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7

    print()
    print("Below is Confusion Matrix for Validation Set")
    print(confusion_matrix(batch_label.cpu(), outputs.cpu()))

    batch_label = torch.empty(0).to(device).float()
    output = torch.empty(0).to(device)

    for (i, batch) in enumerate(test_iter, 1):

        # Getting data for current batch
        batch_input, batch_length = batch.text
        batch_input = batch_input.to(device)
        batch_label = torch.cat((batch_label, batch.label.to(device).float()))

        # Forward step to get prediction
        if model_type == 'rnn' or model_type == 'gru' or model_type == 'lstm':
            output = torch.cat((output, net(batch_input, batch_length)))
        else:
            output = torch.cat((output, net(batch_input)))

    outputs = many_cold(output)

    # Saving model
    if args.save:
        torch.save(net, 'model_' + model_type + '.pt')

    # Confusion Matrix
    print()
    print("Below is Confusion Matrix for Test Set")
    plot_confusion_matrix(batch_label.cpu(), outputs.cpu(), classes=subreddits)
    plt.savefig('model_' + model_type + '_confusion.png')
    plt.show()

    # Plot Losses and Accuracy
    plt.figure()
    plt.plot(plot_epoch, plot_train_loss, label='Training Loss')
    plt.plot(plot_epoch, plot_valid_loss, label='Validation Loss')
    plt.title('Losses as Function of Epoch (' + args.model + ')')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig('model_' + model_type + '_loss.png')
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(plot_epoch, plot_train_acc, label='Training Accuracy')
    plt.plot(plot_epoch, plot_valid_acc, label='Validation Accuracy')
    plt.title('Accuracy as Function of Epoch (' + args.model + ')')
    plt.ylim(0, 1.01)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig('model_' + model_type + '_accuracy.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, choices=['baseline', 'rnn', 'cnn', 'gru', 'lstm'], default='baseline',
                        help="Model type: baseline, rnn, cnn, gru, lstm, (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=100)
    parser.add_argument('--num-class', type=int, default=16)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--tokenizer', type=str, choices=['spacy', 'crazy', 'nltk'], default='nltk')

    args = parser.parse_args()

    main(args)

'''
BASELINE:

--model baseline --lr 0.001 --epochs 100

CNN:

--model cnn --lr 0.001 --epochs 100

RNN:

--model rnn --lr 0.0001 --epochs 100 --rnn-hidden-dim 100 

GRU:

--model gru --lr 0.01 --epochs 100 --rnn-hidden-dim 100 *****
--model gru --lr 0.01 --epochs 100 --rnn-hidden-dim 50 *****
--model gru --lr 0.001 --epochs 100 --rnn-hidden-dim 100

LSTM: 

--model lstm --lr 0.0001 --epochs 100 --rnn-hidden-dim 100 
                 
                 
FOR LARGER DATASETS

GRU:

    n_train = 1500
    n_valid = 200
    n_test = 300
--model gru --lr 0.001 --epochs 150 --rnn-hidden-dim 100

'''
