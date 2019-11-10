import praw
import pandas as pd

from filter import filter

import os

def top_to_csv(subreddit_name, n, reddit):

    # Scrapes the top 'n' posts of 'subreddit' and stores the post title, text body, and number of upvotes
    # into a .csv file.
    # topToCSV requires that the reddit instance is passed in as an argument.
    # topToCSV returns the name of the output file.

    subreddit = reddit.subreddit(subreddit_name)

    top = subreddit.top(limit=n)
    dataset = []
    for submission in top:  # Iterate through each submission to add to a list
            dataset.append(
                (filter(submission.title), filter(submission.selftext), submission.subreddit)
            )

    dataframe = pd.DataFrame(dataset, columns=['title', 'text', 'label'])  # Convert list to DataFrame
    filename = subreddit_name+'_top_'+str(n)+'.csv'  # Output .csv filename
    directory = './dataset'
    if os.path.exists(directory) == False:  # Create 'dataset' folder if it does not already exist
        os.mkdir(directory)
    dataframe.to_csv(path_or_buf=directory+'/'+filename, index=False)

    return filename



def data_split(data):

    # Splits input dataframe into training, validation, and testing sets.

    data_sub = data.loc[data['label'] == 1]
    data_obj = data.loc[data['label'] == 0]
    seed = 10

    # Below, data is split into training, validation, and testing subsets
    train_obj, remaining_obj = train_test_split(data_obj, test_size=0.36, random_state=seed)
    train_sub, remaining_sub = train_test_split(data_sub, test_size=0.36, random_state=seed)
    valid_obj, test_obj = train_test_split(remaining_obj, test_size=0.5555, random_state=seed)
    valid_sub, test_sub = train_test_split(remaining_sub, test_size=0.5555, random_state=seed)

    train = pd.concat((train_obj, train_sub), axis=0)
    valid = pd.concat((valid_obj, valid_sub), axis=0)
    test = pd.concat((test_obj, test_sub), axis=0)

    train.to_csv(path_or_buf='./data/train.csv', sep=',', index=False)
    valid.to_csv(path_or_buf='./data/validation.csv', sep=',', index=False)
    test.to_csv(path_or_buf='./data/test.csv', sep=',', index=False)

    # Below a small number of data is chosen to overfit the model as proof of correctness
    data_sub_over = data_sub.sample(n=25, random_state=seed)
    data_obj_over = data_obj.sample(n=25, random_state=seed)
    overfit = pd.concat((data_obj_over, data_sub_over), axis=0)
    overfit.to_csv(path_or_buf='./data/overfit.csv', sep=',', index=False)

    # Printing out sizes of subsets
    print("Number of Subjective in Train:", train.loc[train["label"] == 1].shape[0])
    print("Number of Objective in Train:", train.loc[train["label"] == 0].shape[0])
    print("Number of Subjective in Valid:", valid.loc[valid["label"] == 1].shape[0])
    print("Number of Objective in Valid:", valid.loc[valid["label"] == 0].shape[0])
    print("Number of Subjective in Test:", test.loc[test["label"] == 1].shape[0])
    print("Number of Objective in Test:", test.loc[test["label"] == 0].shape[0])
    print("Number of Subjective in Overfit:", overfit.loc[overfit["label"] == 1].shape[0])
    print("Number of Objective in Overfit:", overfit.loc[overfit["label"] == 0].shape[0])