import praw
import pandas as pd

from filter import filter

def topToCSV(subreddit_name, n, reddit):

    # Scrapes the top 'n' posts of 'subreddit' and stores the post title, text body, and number of upvotes
    # into a .csv file.
    # topToCSV requires that the reddit instance is passed in as an argument.
    # topToCSV returns the dataframe object.

    subreddit = reddit.subreddit(subreddit_name)

    top = subreddit.top(limit=n)
    dataset = []
    for submission in top:  # Iterate through each submission to add to a list
            dataset.append(
                (filter(submission.title), filter(submission.selftext), submission.score, submission.subreddit)
            )

    dataframe = pd.DataFrame(dataset, columns=['title', 'text', 'upvotes', 'label'])  # Convert list to DataFrame

    dataframe.to_csv(path_or_buf='top_'+str(n)+'_posts_from_'+subreddit_name+'.csv', index=False)

    return dataframe