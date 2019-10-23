import praw

reddit = praw.Reddit(client_id='0bfjHES78X7Fyg',
                     client_secret='yyM87GD70mIP3qRToGuX1F59Sd4',
                     user_agent='SubredditPredictor')

subreddit = reddit.subreddit('jokes')  # Access r/jokes

# Print data about the the sub
print("Top posts from r/" + subreddit.display_name)
print(subreddit.title, "\n")
# print(subreddit.description)


top10 = subreddit.top(limit=10)  # Get the top 10 posts from the sub

for submission in top10:
    print("--------------------------", submission.title, "--------------------------")
    print(submission.selftext, "\n")
    print("     Upvotes:", submission.score, "\n")