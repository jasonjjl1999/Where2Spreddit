from utils import *
from sklearn.utils import shuffle

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

subreddits = [
    'jokes', 'askreddit', 'legaladvice', 'AmItheAsshole', 'tifu', 'todayilearned', 'unpopularopinion',
    'relationship_advice', 'LifeProTips'
]  # List of classes that we will be identifying

label_list = 'Subreddit names vs labels in .csv files: \n \n'

print('Obtained posts from:\n')

n_train = 400  # Number of top posts to load from each subreddit for training
n_valid = 80
n_test = 100

'''
Good for Overfitting:

n_train = 300
n_valid = 50
n_test = 80

Good for training: 

n_train = 600 
n_valid = 100
n_test = 100
'''

n = [n_train, n_valid, n_test]
sample_types = ['train', 'valid', 'test']

for label, subreddit in enumerate(subreddits):  # Create a .csv file for each subreddit
    print(top_to_csv(subreddit, sum(n), label, reddit))
    label_list += 'r/' + subreddit + ': ' + str(label) + '\n'

    #  Store the label/subreddit correspondence in a text file.
    label_file = open('./dataset/labels.txt', 'w')
    label_file.write(label_list)
    label_file.close()

for i, sample_type in enumerate(sample_types):
    # Read all .csv files in './dataset/' directory
    dataset = []
    for filename in os.listdir('./dataset/'):
        if filename.endswith('.csv'):
            sample = pd.read_csv('./dataset/' + filename)
            if sample_type == 'train':
                dataset.append(sample[0:n_train])
            elif sample_type == 'valid':
                dataset.append(sample[n_train:n_train + n_valid])
            elif sample_type == 'test':
                dataset.append(sample[n_train + n_valid:n_train + n_valid + n_test])

    df = pd.concat(dataset, ignore_index=True)  # Concatenate all subreddit data into one dataframe
    df = shuffle(df)

    directory = './dataset/training/'
    if os.path.exists(directory) == False:  # Create 'dataset/training' directory if it does not already exist
        os.mkdir(directory)

    df.to_csv(path_or_buf=directory + '/' + sample_type + '.csv', index=False)

    print(df)
