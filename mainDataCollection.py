from utils import*

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

print('Obtained posts from:')

print(topToCSV('jokes', 100, reddit))
print(topToCSV('askreddit', 100, reddit))
