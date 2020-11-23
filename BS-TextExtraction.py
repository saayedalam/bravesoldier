# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Library for Reddit API
import praw
import pandas as pd

# Reddit API Credentials
reddit = praw.Reddit(client_id='7_PY9asBHeVJxw',
                     client_secret='KL01wgTYZqwEDdPH-R8vNBqFYe4',
                     password='9S2a8a7hcr!',
                     user_agent='bravesoldier by /u/saayed',
                     username='saayed')

# Pull the subreddit of 
subreddit = reddit.subreddit('leaves')

# Pulling top 1000 posts of leaves subreddit
leaves_subreddit = reddit.subreddit('leaves').top(limit=1000)

# Create an empty dictionary to save data
dict = {'title': [],
        'body': [],
       }

# Storing the data in the empty dictionary
for submission in leaves_subreddit:
    dict['title'].append(submission.title)
    dict['body'].append(submission.selftext)

# Convert the data to pandas dataframe and apply date function
df = pd.DataFrame(dict)
df['raw'] = df['title'] + ' ' + df['body']
df.drop(['title', 'body'], axis=1, inplace=True)

# Save it as CSV
df.to_csv('rleaves.csv', index=False)
# -

# !jupytext --to py BS-TextExtraction.ipynb
