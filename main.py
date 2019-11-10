import praw
import pandas as pd

from filter import filter
from utils import*

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

top_jokes = pd.read_csv('dataset/jokes_top_100.csv')
top_askreddit = pd.read_csv('dataset/askreddit_top_100.csv')


data = pd.concat([top_jokes, top_askreddit], ignore_index=True)

print(data)
