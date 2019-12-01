from sklearn.utils import shuffle

from utils import *

if __name__ == '__main__':

    n_train = 50
    n_valid = 20
    n_test = 30

    n = [n_train, n_valid, n_test]
    sample_types = ['train', 'valid', 'test']

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
