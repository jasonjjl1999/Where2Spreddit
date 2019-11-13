from utils import *
from sklearn.utils import shuffle

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

subreddits = [
    'jokes', 'askreddit', 'legaladvice', 'AmItheAsshole', 'tifu', 'todayilearned', 'unpopularopinion',
    'NoStupidQuestions', 'relationship_advice', 'LifeProTips'
]  # List of classes that we will be identifying

label_list = 'Subreddit names vs labels in .csv files: \n \n'

print('Obtained posts from:\n')

n_train = 300  # Number of top posts to load from each subreddit for training
n_valid = 100
n_test = 50

n = [n_train, n_valid, n_test]
sample_types = ['train', 'valid', 'test']

for i, sample_type in enumerate(sample_types):
    for label, subreddit in enumerate(subreddits):  # Create a .csv file for each subreddit
        print(top_to_csv(subreddit, n[i], label, reddit, sample_type))
        label_list += 'r/' + subreddit + ': ' + str(label) + '\n'

    #  Store the label/subreddit correspondence in a text file.
    if i == 0:  # Write the text file only once
        label_file = open('./dataset/labels.txt', 'w')
        label_file.write(label_list)
        label_file.close()

    # Read all .csv files in './dataset/<sample_type>' directory
    dataset = []
    for filename in os.listdir('./dataset/'+sample_type+'/'):
        if filename.endswith('.csv'):
            dataset.append(pd.read_csv('./dataset/'+sample_type+'/'+ filename))

    df = pd.concat(dataset, ignore_index=True)  # Concatenate all subreddit data into one dataframe
    df = shuffle(df)

    directory = './dataset/'+sample_type
    if os.path.exists(directory) == False:  # Create 'dataset/training' directory if it does not already exist
        os.mkdir(directory)

    df.to_csv(path_or_buf=directory + '/'+sample_type+'.csv', index=False)

    print(df)


