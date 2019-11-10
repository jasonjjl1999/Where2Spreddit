import praw
import pandas as pd

from filter import filter

import os

def topToCSV(subreddit_name, n, reddit):

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