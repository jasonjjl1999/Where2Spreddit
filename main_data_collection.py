from utils import*

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')



subreddits = [
    'jokes', 'askreddit', 'legaladvice', 'AmItheAsshole', 'tifu', 'todayilearned', 'unpopularopinion',
    'NoStupidQuestions', 'relationship_advice', 'LifeProTips'
]

label_list = 'Subreddit names vs labels in .csv files: \n \n'

print('Obtained posts from:\n')

n = 300  # Number of top posts to load from each subreddit

for label, subreddit in enumerate(subreddits):  # Create a .csv file for each subreddit
    print(top_to_csv(subreddit, n, label, reddit))
    label_list += subreddit+': '+str(label)+'\n'

#  Store the label/subreddit correspondence in a text file.
label_file = open('./dataset/labels.txt', 'w')
label_file.write(label_list)
label_file.close()

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