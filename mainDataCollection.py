from utils import*

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

print(topToCSV('jokes', 100, reddit))

x = pd.read_csv('jokes_top_100.csv')
