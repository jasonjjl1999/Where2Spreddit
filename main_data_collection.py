from utils import*

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

n = 500  # Number of top posts to load from each subreddit

print('Obtained posts from:\n')

print(top_to_csv('jokes', n, reddit))
print(top_to_csv('askreddit', n, reddit))
print(top_to_csv('legaladvice', n, reddit))
print(top_to_csv('AmItheAsshole', n, reddit))
print(top_to_csv('tifu', n, reddit))
print(top_to_csv('todayilearned', n, reddit))
print(top_to_csv('unpopularopinion', n, reddit))
print(top_to_csv('NoStupidQuestions', n, reddit))
print(top_to_csv('relationship_advice', n, reddit))
print(top_to_csv('LifeProTips', n, reddit))



# Read all .csv files in './dataset' directory
dataset = []
for filename in os.listdir('./dataset'):
    if filename.endswith('.csv'):
        dataset.append(pd.read_csv('./dataset/'+filename))

df = pd.concat(dataset, ignore_index=True)  # Concatenate all subreddit data into one dataframe

directory = './dataset/training'
if os.path.exists(directory) == False:  # Create 'dataset/training' directory if it does not already exist
    os.mkdir(directory)

df.to_csv(path_or_buf=directory+'/df.csv', index=False)

print(df)

data_split(df)