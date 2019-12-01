from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def filter(post):
    # Lowercase
    post = post.lower()

    # Contractions
    # List from https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
    post = post.replace("ain't", "is not")
    post = post.replace("aren't", "are not")
    post = post.replace("can't", "cannot")
    post = post.replace("could've", "could have")
    post = post.replace("didn't", "did not")
    post = post.replace("doesn't", "does not")
    post = post.replace("don't", "do not")
    post = post.replace("gimme", "give me")
    post = post.replace("gotta", "got to")
    post = post.replace("hadn't", "had not")
    post = post.replace("hasn't", "has not")
    post = post.replace("haven't", "have not")
    post = post.replace("i'll", "i will")
    post = post.replace("let's", "let us")
    post = post.replace("they're", "they are")
    post = post.replace("they've", "they have")
    post = post.replace("wasn't", "was not")
    post = post.replace("we'll", "we will")
    post = post.replace("we're", "we are")
    post = post.replace("we've", "we have")
    post = post.replace("weren't", "were not")
    post = post.replace("what's", "what is")
    post = post.replace("when's", "when is")
    post = post.replace("where's", "where is")
    post = post.replace("won't", "will not")
    post = post.replace("would've", "would have")
    post = post.replace("you're", "you are")

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
