from utils import*

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

print('Obtained posts from:')
print(topToCSV('jokes', 1000, reddit))
print(topToCSV('askreddit', 1000, reddit))
print(topToCSV('legaladvice', 1000, reddit))

# Read all .csv files in './dataset' directory
dataset = []
for filename in os.listdir('./dataset'):
    if filename.endswith('.csv'):
        dataset.append(pd.read_csv('./dataset/'+filename))

df = pd.concat(dataset, ignore_index=True)  # Concatenate all subreddit data into one dataframe
df.to_csv(path_or_buf='./dataset/df.csv', index=False)
