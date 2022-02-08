import os
import json
import csv
import time
import datetime
import tweepy  # note that API is 1.1 interface. We will use 1.2 Interface because 1.1 will eventually be deprecated.
import pandas as pd
import ast
import pickle


class multi_clienthandler:
    poll_fields = ["id", "options", "duration_minutes", "end_datetime", "voting_status"]
    place_fields = ["full_name", "id", "contained_within", "country", "country_code", "geo", "name", "place_type"]
    tweet_fields = ["id", "text", "attachments", "author_id", "context_annotations", "conversation_id", "created_at",
                    "entities", "geo", "in_reply_to_user_id", "lang", "possibly_sensitive", "public_metrics",
                    "referenced_tweets", "reply_settings", "source", "withheld"]
    # organic_metrics, non_public_metrics, promoted_metrics
    # are left out. They require user context authentication
    media_fields = ["media_key", "type", "duration_ms", "height", "preview_image_url", "public_metrics", "width", "url"]
    # again, non_public_metrics, promoted_metrics, organic_metrics is left out

    user_fields = ["id", "name", "username", "created_at", "description", "entities", "location", "pinned_tweet_id",
                   "profile_image_url", "protected", "public_metrics", "url", "verified", "withheld"]
    expansions = ["author_id", "referenced_tweets.id", "referenced_tweets.id.author_id", "entities.mentions.username",
                  "attachments.poll_ids", "attachments.media_keys", "in_reply_to_user_id", "geo.place_id"]

    def __init__(self, appkeylist, bearerslist, global_pause=10, will_pull_media=False, global_max_results=100):
        self.clientlist = []
        self.appkeylist = appkeylist
        self.global_pause = global_pause
        self.bearerslist = bearerslist
        self.will_pull_media = will_pull_media
        self.global_max_results = global_max_results

    def get_user_mentions(self, userid, starttime="2006-03-30T00:01:00-08:00", endtime=(
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=20)).astimezone().isoformat(),
                          requested_number=100, savedToken=None, user_auth=False):
        # i'm not sure why but i'm always unauthorized...
        # it looks like academic doesn't stop too many requests from triggering on this almost instantly
        # anyway it might be locked to the user itself only. it's not documented...
        datetimetype = type(datetime.datetime.now())
        tweepyreferenced_tweet = type(tweepy.tweet.ReferencedTweet({"id": 0, "type": 0}))
        force_end = False
        max_results = self.global_max_results
        saved_data = []
        # max tweets retrievable for this is.. 800.
        current_bearer = 0
        while True:
            try:
                if savedToken:
                    obtained_result = self.appkeylist[current_bearer].get_users_mentions(id=userid,
                                                                                         start_time=starttime,
                                                                                         end_time=endtime,
                                                                                         pagination_token=savedToken,
                                                                                         max_results=max_results,
                                                                                         user_auth=user_auth,
                                                                                         tweet_fields=self.tweet_fields)  # expansions=self.expansions,media_fields=self.media_fields,poll_fields=self.poll_fields,user_fields=self.user_fields)
                    # what's the likelihood you including user auth lets you get that privated account tweet? really small.
                    # let's just leave an option anyway. Keep in mind said API key must be user logged in, and because we cycle, all must be user logged in.

                else:
                    obtained_result = self.appkeylist[current_bearer].get_users_mentions(id=userid,
                                                                                         start_time=starttime,
                                                                                         end_time=endtime,
                                                                                         pagination_token=savedToken,
                                                                                         max_results=max_results,
                                                                                         user_auth=user_auth,
                                                                                         tweet_fields=self.tweet_fields)  # expansions=self.expansions,media_fields=self.media_fields,poll_fields=self.poll_fields,user_fields=self.user_fields)
                if obtained_result.data:
                    saved_data.extend(obtained_result.data)
                else:
                    print("failed. No tweets with this query.")

                print("pulled:", len(obtained_result.data), "tweets.")
                try:
                    savedToken = obtained_result.meta["next_token"]
                except KeyError:
                    savedToken = None
                    force_end = True
                if len(saved_data) >= requested_number or force_end:
                    for i in range(len(saved_data)):
                        saved_data[i] = {j__: getattr(saved_data[i], j__, None) for j__ in
                                         saved_data[i].__slots__}  # convert to dictionary.
                        for keyitem in saved_data[i]:
                            if type(saved_data[i][keyitem]) == datetimetype:
                                saved_data[i][keyitem] = saved_data[i][keyitem].isoformat()
                            elif type(saved_data[i][keyitem]) == type([]):
                                for item_num in range(len(saved_data[i][keyitem])):
                                    if type(saved_data[i][keyitem][item_num]) == tweepyreferenced_tweet:
                                        saved_data[i][keyitem][item_num] = {
                                            j__: getattr(saved_data[i][keyitem][item_num], j__, None) for j__ in
                                            saved_data[i][keyitem][item_num].__slots__}  # convert to dictionary.
                    print("pulls completed. returning")

                    return saved_data, savedToken
                time.sleep(2)
            except tweepy.errors.TwitterServerError as e:
                time.sleep(1)
                continue
            except (
            tweepy.errors.BadRequest, tweepy.errors.Unauthorized, tweepy.errors.Forbidden, tweepy.errors.NotFound) as e:
                raise ValueError(e)
            except tweepy.errors.TooManyRequests as e:
                time.sleep(1)
                current_bearer += 1
                if current_bearer >= len(self.bearerslist):
                    # print(e)
                    print("sleeping for ", self.global_pause, " seconds, due to TooManyRequests Error.")
                    time.sleep(self.global_pause)
                    current_bearer = 0

    def get_tweet(self, tweetid):
        datetimetype = type(datetime.datetime.now())
        tweepyreferenced_tweet = type(tweepy.tweet.ReferencedTweet({"id": 0, "type": 0}))

        current_bearer = 0
        totaltries = 0
        while True:
            try:
                obtained_result = self.appkeylist[current_bearer].get_users_tweets(id=tweetid, user_auth=False,
                                                                                   tweet_fields=self.tweet_fields)
                saved_data = [obtained_result]
                for i in range(len(saved_data)):
                    saved_data[i] = {j__: getattr(saved_data[i], j__, None) for j__ in
                                     saved_data[i].__slots__}  # convert to dictionary.
                    for keyitem in saved_data[i]:
                        if type(saved_data[i][keyitem]) == datetimetype:
                            saved_data[i][keyitem] = saved_data[i][keyitem].isoformat()
                        elif type(saved_data[i][keyitem]) == type([]):
                            for item_num in range(len(saved_data[i][keyitem])):
                                if type(saved_data[i][keyitem][item_num]) == tweepyreferenced_tweet:
                                    saved_data[i][keyitem][item_num] = {
                                        j__: getattr(saved_data[i][keyitem][item_num], j__, None) for j__ in
                                        saved_data[i][keyitem][item_num].__slots__}  # convert to dictionary.

                return saved_data[0]
            except:
                current_bearer += 1
                if current_bearer > len(self.appkeylist):
                    current_bearer = 0
                    totaltries += 1
                    if totaltries > 3:
                        return None

    def get_users_tweets(self, tweetid, starttime="2006-03-30T00:01:00-08:00", endtime=(
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=20)).astimezone().isoformat(),
                         requested_number=100, savedToken=None, user_auth=False):
        max_results = self.global_max_results
        saved_data = []
        current_bearer = 0
        datetimetype = type(datetime.datetime.now())
        tweepyreferenced_tweet = type(tweepy.tweet.ReferencedTweet({"id": 0, "type": 0}))
        force_end = False

        while True:
            try:
                if savedToken:
                    obtained_result = self.appkeylist[current_bearer].get_users_tweets(id=tweetid, start_time=starttime,
                                                                                       end_time=endtime,
                                                                                       pagination_token=savedToken,
                                                                                       max_results=max_results,
                                                                                       user_auth=user_auth,
                                                                                       tweet_fields=self.tweet_fields)  # expansions=self.expansions,media_fields=self.media_fields,poll_fields=self.poll_fields,user_fields=self.user_fields)
                    # what's the likelihood you including user auth lets you get that privated account tweet? really small.
                    # let's just leave an option anyway. Keep in mind said API key must be user logged in, and because we cycle, all must be user logged in.

                else:
                    obtained_result = self.appkeylist[current_bearer].get_users_tweets(id=tweetid, start_time=starttime,
                                                                                       end_time=endtime,
                                                                                       max_results=max_results,
                                                                                       user_auth=user_auth,
                                                                                       tweet_fields=self.tweet_fields)  # expansions=self.expansions,media_fields=self.media_fields,poll_fields=self.poll_fields,user_fields=self.user_fields,place_fields=self.place_fields)
                if obtained_result.data:
                    saved_data.extend(obtained_result.data)
                else:
                    print("failed. No tweets with this query.")
                print("pulled:", len(obtained_result.data), "tweets.")
                try:
                    savedToken = obtained_result.meta["next_token"]
                except KeyError:
                    savedToken = None
                    force_end = True
                if len(saved_data) >= requested_number or force_end:
                    for i in range(len(saved_data)):
                        saved_data[i] = {j__: getattr(saved_data[i], j__, None) for j__ in
                                         saved_data[i].__slots__}  # convert to dictionary.
                        for keyitem in saved_data[i]:
                            if type(saved_data[i][keyitem]) == datetimetype:
                                saved_data[i][keyitem] = saved_data[i][keyitem].isoformat()
                            elif type(saved_data[i][keyitem]) == type([]):
                                for item_num in range(len(saved_data[i][keyitem])):
                                    if type(saved_data[i][keyitem][item_num]) == tweepyreferenced_tweet:
                                        saved_data[i][keyitem][item_num] = {
                                            j__: getattr(saved_data[i][keyitem][item_num], j__, None) for j__ in
                                            saved_data[i][keyitem][
                                                item_num].__slots__}  # convert to dictionary. # convert to dictionary.
                    print("pulls completed. returning")
                    return saved_data, savedToken
                time.sleep(2)
            except tweepy.errors.TwitterServerError as e:
                time.sleep(1)
                continue
            except (
            tweepy.errors.BadRequest, tweepy.errors.Unauthorized, tweepy.errors.Forbidden, tweepy.errors.NotFound) as e:
                raise ValueError(e)
            except tweepy.errors.TooManyRequests as e:
                time.sleep(1)
                current_bearer += 1
                if current_bearer >= len(self.bearerslist):
                    # print(e)
                    print("sleeping for ", self.global_pause, " seconds, due to TooManyRequests Error.")
                    time.sleep(self.global_pause)
                    current_bearer = 0

    def archive_search(self, query, starttime="2006-03-30T00:01:00-08:00", endtime=(
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=20)).astimezone().isoformat(),
                       additional_directory="", requested_number=100, savedToken=None):
        max_results = self.global_max_results
        # failure to specify results in searching from the beginning of twitter to around 20 seconds from now.
        current_bearer = 0
        print(current_bearer)
        saved_data = []
        datetimetype = type(datetime.datetime.now())
        tweepyreferenced_tweet = type(tweepy.tweet.ReferencedTweet({"id": 0, "type": 0}))
        if not os.path.exists(additional_directory):  # REMOVE MY ARGUMENT TOOO
            os.mkdir(additional_directory)

        dumpcount = 0

        force_end = False
        while True:
            try:
                if savedToken:
                    obtained_result = self.bearerslist[current_bearer].search_all_tweets(query, start_time=starttime,
                                                                                         end_time=endtime,
                                                                                         expansions=self.expansions,
                                                                                         max_results=max_results,
                                                                                         next_token=savedToken,
                                                                                         poll_fields=self.poll_fields,
                                                                                         place_fields=self.place_fields,
                                                                                         tweet_fields=self.tweet_fields,
                                                                                         media_fields=self.media_fields)
                else:
                    obtained_result = self.bearerslist[current_bearer].search_all_tweets(query, start_time=starttime,
                                                                                         end_time=endtime,
                                                                                         expansions=self.expansions,
                                                                                         max_results=max_results,
                                                                                         poll_fields=self.poll_fields,
                                                                                         place_fields=self.place_fields,
                                                                                         tweet_fields=self.tweet_fields,
                                                                                         media_fields=self.media_fields)
                if obtained_result.data:
                    saved_data.extend(obtained_result.data)
                    print("pulled:", len(obtained_result.data), "tweets.")
                else:
                    print("failed. No tweets with this query.")
                    return None, None, 0

                dumpcount += len(obtained_result.data)  # delete me
                try:
                    savedToken = obtained_result.meta["next_token"]
                except KeyError:
                    savedToken = None
                    force_end = True
                print("-" * 30, "sample", "-" * 30)
                print(obtained_result.data[0]["text"])
                print(savedToken)
                print("-" * 75)

                if len(saved_data) >= requested_number or force_end:
                    # if len(saved_data)>=requested_number or force_end:
                    for i in range(len(saved_data)):
                        saved_data[i] = {j__: getattr(saved_data[i], j__, None) for j__ in
                                         saved_data[i].__slots__}  # convert to dictionary.
                        for keyitem in saved_data[i]:
                            if type(saved_data[i][keyitem]) == datetimetype:
                                saved_data[i][keyitem] = saved_data[i][keyitem].isoformat()
                            elif type(saved_data[i][keyitem]) == type([]):
                                for item_num in range(len(saved_data[i][keyitem])):
                                    if type(saved_data[i][keyitem][item_num]) == tweepyreferenced_tweet:
                                        saved_data[i][keyitem][item_num] = {
                                            j__: getattr(saved_data[i][keyitem][item_num], j__, None) for j__ in
                                            saved_data[i][keyitem][item_num].__slots__}  # convert to dictionary.

                    # with open(os.path.join(additional_directory,query+"_"+"_".join(starttime.split("-")[:2])+".json"),"w",encoding="utf-8") as milkteacheck:
                    # note: Dan here, edited this for myself
                    # TODO unfuck this mess later
                    with open(os.path.join(additional_directory, query.split(":")[1] + ".json"), "w",
                              encoding="utf-8") as milkteacheck:
                        json.dump(saved_data, milkteacheck, indent=4)  # remove
                        saved_data.clear()
                        print("pulls completed. returning")  # unindent us
                        return saved_data, savedToken, len(obtained_result.data)  # unindent us
                time.sleep(1)

            except tweepy.errors.TwitterServerError as e:
                time.sleep(1)
                continue

            except (
            tweepy.errors.BadRequest, tweepy.errors.Unauthorized, tweepy.errors.Forbidden, tweepy.errors.NotFound) as e:

                raise ValueError(e)
            except tweepy.errors.TooManyRequests as e:
                time.sleep(1)
                current_bearer += 1
                if current_bearer >= len(self.bearerslist):
                    # print(e)
                    print("sleeping for ", self.global_pause, " seconds, due to TooManyRequests Error.")
                    time.sleep(self.global_pause)
                    current_bearer = 0

    # BadRequest(HTTPException) """Exception raised for a 400 HTTP status code"""
    # Unauthorized(HTTPException) """Exception raised for a 401 HTTP status code"""
    # Forbidden(HTTPException) """Exception raised for a 403 HTTP status code"""
    # NotFound(HTTPException) """Exception raised for a 404 HTTP status code"""
    # TooManyRequests(HTTPException) """Exception raised for a 429 HTTP status code"""
    # TwitterServerError(HTTPException) """Exception raised for a 5xx HTTP status code"""


if __name__ == "__main__":

    # app_key_1 = ""
    # app_secret_1 = ""
    # [[app_key_1, app_secret_1],blahblah]

    app_key_2 = "WGxzGBtTzvY8l1zMotHFQN4Dm"
    app_secret_2 = "ODFT6kl3cJKJzO1MtddW8iU6IjvbYKJWRbfA5X7WgvpnKF850Q"

    app_key_3 = "qNQjmEzhH5FLikYvsdunV6eeT"
    app_secret_3 = "ItHU9TF9h1hj4jyYJU9FGb9sNJXP05kvt6hbSTk5IDvvrEC0Ll"

    app_key_4 = "fE1fzFW2KGfKF68jzDnZBSoER"
    app_secret_4 = "c9vAqxdOpjUc4ZPvHorcCP1Fk7ZCsJ2qqXR6UeBuJxYgBmraem"

    app_key_5 = "DrDuOqf8wS3MqLwfP8g8PX3ma"
    app_secret_5 = "O56eDLTKL5Pb0ryKfocMz5IWTeHU1honmGd9IM8dx2A9hOD2eP"

    bearer_token_list = [
        "AAAAAAAAAAAAAAAAAAAAAA2KMwEAAAAA9ZsLMnP3vIkF5n8pCSFhmDrVccY%3D32PZo4XDLsICrS0OGrAVVjcjBGqx06rxcJ461Tar7el0ZmnBVa"]

    handle_keys = [[app_key_2, app_secret_2], [app_key_3, app_secret_3], [app_key_4, app_secret_4],
                   [app_key_5, app_secret_5]]
    parser = tweepy.parsers.JSONParser()
    app_key_secret_clients = []
    apiv2_bearer_clients = []

    for k in handle_keys:
        app_key_secret_clients.append(
            tweepy.client.Client(access_token=k[0], access_token_secret=k[1], wait_on_rate_limit=False))
    for k in bearer_token_list:
        apiv2_bearer_clients.append(tweepy.client.Client(bearer_token=k, wait_on_rate_limit=False))

    instance = multi_clienthandler(app_key_secret_clients, apiv2_bearer_clients, global_pause=10,
                                   global_max_results=100)

    # keywords_list = [[["2020-04-01T00:01:00-08:00","2020-05-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-05-01T00:01:00-08:00","2020-06-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-06-01T00:01:00-08:00","2020-07-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-07-01T00:01:00-08:00","2020-08-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-08-01T00:01:00-08:00","2020-09-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-09-01T00:01:00-08:00","2020-10-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-10-01T00:01:00-08:00","2020-11-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-11-01T00:01:00-08:00","2020-12-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2020-12-01T00:01:00-08:00","2021-01-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2021-01-01T00:01:00-08:00","2021-02-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2021-02-01T00:01:00-08:00","2021-03-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2021-03-01T00:01:00-08:00","2021-04-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2021-04-01T00:01:00-08:00","2021-05-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]],
    # [["2021-05-01T00:01:00-08:00","2021-06-01T00:01:00-08:00","MILKTEA_SEGREGATED"],["#MilkTeaAlliance"]]
    # ]

    '''
    The real usecase, opening csv with keywords
    '''

    search_list = []
    no_of_results = []

    # df = pd.read_csv("tweets_counted_full_usethis.tsv", sep='\t', )
    # df = df.loc[df['results_count'] != 0]
    # print("number of articles: ")
    # print(len(df))
    # print (df.iloc[98:300])

    root_dir = "viet_raw_results"
    for file in os.listdir(root_dir):
        with open(os.path.join(root_dir, file), 'r') as f:
            content = json.load(f)
        convo_id_list = []
        for i in content:
            convo_id = i['conversation_id']
            convo_id_list.append("conversation_id:{}".format(convo_id))

        filename = file.split(".")[0]

        search_list.append([filename, convo_id_list])

    print(search_list)


    # for index, row in df.iloc[100: 300].iterrows():
    #     post_time = datetime.datetime.strptime(row['date'].strip(), '%d/%m/%Y')
    #     # search within 4 weeks period
    #     start_date = post_time - datetime.timedelta(days=28)
    #     end_date = post_time + datetime.timedelta(days=28)
    #
    #     start_date_str = start_date.replace(tzinfo=datetime.timezone.utc).replace(microsecond=0).isoformat()
    #     end_date_str = end_date.replace(tzinfo=datetime.timezone.utc).replace(microsecond=0).isoformat()
    #
    #     keywords = ast.literal_eval(row['keywords_eng'])
    #     keywords_str = " ".join(keywords)
    #     # print(keywords_str)

        # data_path = os.path.join("indo_search_results",
        #                          keywords_str.strip() + "_" + "_".join(start_date_str.split("-")[:2]) + ".json")
        # if os.path.exists(data_path):
        #     with open(data_path, 'r') as f:
        #         content = json.load(f)
        #         convo_id_list = []
        #         # print(content)
        #         for i in content:
        #             convo_id = i['conversation_id']
        #             convo_id_list.append("conversation_id:{}".format(convo_id))

        # search_list.append([keywords_str, convo_id_list])

    print(search_list)
    # print(start_date, end_date)

    # keywords_list = [[["2020-04-01T00:01:00-08:00", "2021-11-05T00:01:00-08:00", "test_convo"], ["conversation_id:{}".format(convo_id)]]]

    # [" DATE TIME HERE START", " DATE TIME HERE END", "WHATEVER FILE YOU WANNA SAVE IT IN. WILL CREATE FOR YOU OWO"],["WHATEVER SEARCH TERM"]  <- THIS IS 1 QUERY
    # FORMAT TOGETHER IN A LIST LIKE YOU SEE ABOVE
    print("pulling", len(search_list), " events.")
    if not os.path.exists("convo_search_results"):
        os.mkdir("convo_search_results")

    for i in search_list:
        path_to_save = os.path.join("convo_search_results", i[0])

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        filelist = os.listdir(path_to_save)
        for hashtag in i[1]:
            print("attempting: ", hashtag)
            # print(os.path.join("MILKTEA_SEGREGATED","#MilkTeaAlliance"+"_"+"_".join(i[0][0].split("-")[:2])+".json"))
            print(os.path.join(path_to_save, hashtag.split(":")[1] + ".json"))
            # if os.path.exists(os.path.join(path_to_save, hashtag.split(":")[1] + ".json")):
            #     print(os.path.join(path_to_save, hashtag.split(":")[1] + ".json"), "was done..")
            #     # no_of_results.append(0)
            #     continue
            if os.path.exists(os.path.join(path_to_save, hashtag.split(":")[1] + ".json")):
                print(os.path.join(path_to_save, hashtag.split(":")[1] + ".json"), "was done..")
                # no_of_results.append(0)
                continue
            # print(i)
            else:
                output, savedToken, number_of_tweets = instance.archive_search(hashtag, requested_number=100,
                                                                               additional_directory=path_to_save)
                print("length of output: {}".format(number_of_tweets))
            # no_of_results.append(number_of_tweets)

    # saved token is used to move to the next "page" of tweets retrievable.
    # YYYY-MM-DDTHH:mm:ssZ

    # print(instance.get_user_mentions(id="759251",requested_number=10,savedToken=None,user_auth=False))
    # untestable... you need user auth probably...

    # output,_ =instance.get_users_tweets("759251",starttime="2006-03-30T00:01:00-08:00",endtime=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=20)).astimezone().isoformat(),requested_number=100,savedToken=None,user_auth=False)
    # DOES NOT WORK!!! they haven't even introduced this.

    # just in case of a screw up
    with open('number_of_results_list.pickle', 'wb') as f:
        pickle.dump(no_of_results, f)

    df['number_of_results'] = no_of_results
    df.to_csv("indo_with_tweet_counts", index=False, sep='\t')


