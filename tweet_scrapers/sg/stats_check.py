import os
import json

threads_count = 0
tweets_count = 0
for article in os.listdir("root_tweets_results"):
    threads_count += len(os.listdir(os.path.join("root_tweets_results", article)))
    for f in os.listdir(os.path.join("root_tweets_results", article)):
        with open(os.path.join("root_tweets_results", article, f), "r") as handle:
            content = json.load(handle)
            tweets_count += len(content)

# for article in os.listdir("root_tweets_results"):
#     threads_count += len(os.listdir(os.path.join("root_tweets_results", article)))

print(threads_count)
print(tweets_count)
# print(len(os.listdir("root_tweets_results")))