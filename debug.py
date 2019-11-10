import praw
import pandas as pd

from filter import filter

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

subreddit = reddit.subreddit('jokes')  # Access r/jokes

# Print data about the the sub
print('Top posts from r/' + subreddit.display_name)
print(subreddit.title, '\n')
# print(subreddit.description)


top = subreddit.top(limit=100)  # Get the top 100 posts from the sub

dataset = []

for submission in top:

    # Print out details about the submission
    print('--------------------------', submission.title, '--------------------------')
    print(submission.selftext, '\n')
    print('     Upvotes:', submission.score, '\n')

    # Add the submission to a Pandas DataFrame object
    if submission.selftext != '':  # Filters out non text posts
        dataset.append((filter(submission.title), filter(submission.selftext), submission.score, submission.subreddit))

dataset = pd.DataFrame(dataset, columns=['title', 'text', 'upvotes', 'label'])

print('The pandas DataFrame of the data: ')
print(dataset)

dataset.to_csv(path_or_buf='dataset.csv', index=False)

