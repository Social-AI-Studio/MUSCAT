from keybert import KeyBERT
import os
import json
import pandas as pd
import html

keywords_list = []
kw_model = KeyBERT()

rootdir = "blackdot_cleaned"
for file in os.listdir(rootdir):
    with open(os.path.join(rootdir, file), 'r') as f:
        content = json.load(f)
        title = content['Title']
        # print(title)
        keywords = kw_model.extract_keywords(title, stop_words="english", keyphrase_ngram_range=(1, 1), top_n=5)
        sorted_keywords = sorted(keywords, key=lambda tup: tup[1], reverse=True)
        # print(sorted_keywords)
        selected_keywords = [i[0] for i in keywords]

        #remove these keywords cause it's unlikely to help
        selected_keywords = [i for i in selected_keywords if i not in ("covidwatch", "vaccinewatch")]
        print(file)
        print(selected_keywords)

        content["keywords"] = selected_keywords
        path_to_save = rootdir+"_with_keywords"
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        with open(os.path.join(path_to_save, file), 'w') as handle:
            json.dump(content, handle, indent=4)
# df = pd.read_csv("indo_new_trans.csv", encoding='utf-8')
# print(df)
#
# titles_eng = [html.unescape(i) for i in df["Title_Eng"]]
#
# for title in titles_eng:
#     # use unescape to fix the html entities
#     print(html.unescape(title))
#     keywords = kw_model.extract_keywords(html.unescape(title), stop_words="english", keyphrase_ngram_range=(1, 1))
#     # only pick keywords with a confidence score above 0.1
#     selected_keywords = [i[0] for i in keywords]
#     print(selected_keywords)
#     keywords_list.append(selected_keywords)
#
# df["keywords"] = keywords_list
# df.to_csv("indo_with_keywords.csv", index=False)

