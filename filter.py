from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def filter(post):



    # Lowercase
    post = post.lower()

    # LifeProTips prefix
    post = post.replace('lpt', '')

    # tifu prefix
    post = post.replace('tifu', '')


    # todayilearned prefix
    post = post.replace('til', '')

    # Tokenize and Stemming: https://machinelearningmastery.com/clean-text-machine-learning-python/
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(post)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if (word.isalpha() or word == '.' or word == '?')]

    # Convert back into string
    post = ' '.join(tokens)

    return post
