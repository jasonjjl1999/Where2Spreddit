from utils import *


from psaw import PushshiftAPI
import praw

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

api = PushshiftAPI(reddit)

subreddits = [
    'jokes', 'askreddit', 'legaladvice', 'tifu', 'todayilearned', 'technology',
    'relationship_advice', 'LifeProTips', 'askscience', 'personalfinance', 'movies', 'science',
    'worldnews', 'history', 'space', 'sports'
]  # List of classes that we will be identifying

if __name__ == '__main__':  # Do not recollect data on import

    n = 100

    label_list = 'Subreddit names vs labels in .csv files: \n \n'

    print('Obtained posts from:\n')

    for label, subreddit in enumerate(subreddits):  # Create a .csv file for each subreddit
        print(top_to_csv(subreddit, n, label, reddit, api))  # Take 100 extra posts in case some are not text-based
        label_list += 'r/' + subreddit + ': ' + str(label) + '\n'

        #  Store the label/subreddit correspondence in a text file.
        label_file = open('./dataset/labels.txt', 'w')
        label_file.write(label_list)
        label_file.close()
