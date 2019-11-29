def filter(post):
    # Replace all 'curly' apostrophes and quotes
    post = post.replace('“', '"')
    post = post.replace('”', '"')
    post = post.replace("‘", "'")
    post = post.replace("’", "'")

    # LifeProTips prefix
    post = post.replace('LPT: ', '')
    post = post.replace('LPT : ', '')
    post = post.replace('Lpt: ', '')

    # AmItheAsshole prefix
    post = post.replace('AITA ', '')
    post = post.replace('WIBTA ', '')

    # tifu prefix
    post = post.replace('TIFU: ', '')
    post = post.replace('TIFU ', '')

    # todayilearned prefix
    post = post.replace('TIL: ', '')
    post = post.replace('TIL ', '')

    # Square brackets for tags
    post = post.replace('[', '')
    post = post.replace(']', '')

    # Remove accents
    post = post.replace('é', 'e')

    # Change Signs
    post = post.replace('€', 'dollars')
    post = post.replace('$', 'dollars')

    # Lowercase
    post = post.lower()

    return post
