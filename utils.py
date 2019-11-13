import praw
import pandas as pd

from filter import filter

import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np

def top_to_csv(subreddit_name, n, label, reddit):

    # Scrapes the top 'n' posts of 'subreddit' and stores the post title, text body, and number of upvotes
    # into a .csv file in the directory './dataset/'
    # The label is an input int that will be used to label posts from this subreddit
    # topToCSV requires that the reddit instance is passed in as an argument.
    # topToCSV returns the name of the output file.

    subreddit = reddit.subreddit(subreddit_name)

    top = subreddit.top(limit=n)
    dataset = []
    for submission in top:  # Iterate through each submission to add to a list
            dataset.append(
                # (filter(submission.title), label)
                (filter(submission.title)+'. '+filter(submission.selftext), label)
            )

    dataframe = pd.DataFrame(dataset, columns=['text', 'label'])  # Convert list to DataFrame

    '''
    # Same thing, but have titles and body text separate
    for submission in top:  # Iterate through each submission to add to a list
            dataset.append(
                (filter(submission.title), filter(submission.selftext), submission.subreddit)
            )
    dataframe = pd.DataFrame(dataset, columns=['title', 'text', 'label'])  
    '''
    filename = subreddit_name+'_top_'+str(n)+'.csv'  # Output .csv filename
    directory = './dataset/'

    if os.path.exists(directory) == False:  # Create 'dataset' directory if it does not already exist
        os.mkdir(directory)
    dataframe.to_csv(path_or_buf=directory+'/'+filename, index=False)

    return filename



def data_split(data):

    # Shuffle dataset
    seed = 10
    np.random.seed(seed)
    data = shuffle(data)

    # Below, data is split into training, validation, and testing subsets
    train, remaining = train_test_split(data, test_size=0.36, random_state=seed)

    valid, test = train_test_split(remaining, test_size=0.5555, random_state=seed)

    train.to_csv(path_or_buf='./dataset/training/train.csv', sep=',', index=False)
    valid.to_csv(path_or_buf='./dataset/training/validation.csv', sep=',', index=False)
    test.to_csv(path_or_buf='./dataset/training/test.csv', sep=',', index=False)

    # Below a small number of data is chosen to overfit the model as proof of correctness
    overfit = data.sample(n=200, random_state=seed)
    overfit.to_csv(path_or_buf='./dataset/training/overfit.csv', sep=',', index=False)


    # Printing out sizes of subsets
    '''
    print("Number of Subjective in Train:", train.loc[train["label"] == 1].shape[0])
    print("Number of Objective in Train:", train.loc[train["label"] == 0].shape[0])
    print("Number of Subjective in Valid:", valid.loc[valid["label"] == 1].shape[0])
    print("Number of Objective in Valid:", valid.loc[valid["label"] == 0].shape[0])
    print("Number of Subjective in Test:", test.loc[test["label"] == 1].shape[0])
    print("Number of Objective in Test:", test.loc[test["label"] == 0].shape[0])
    print("Number of Subjective in Overfit:", overfit.loc[overfit["label"] == 1].shape[0])
    print("Number of Objective in Overfit:", overfit.loc[overfit["label"] == 0].shape[0])
    '''