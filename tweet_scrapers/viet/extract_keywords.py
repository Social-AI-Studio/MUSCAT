from keybert import KeyBERT
import os
import json
import pandas as pd
import html
from translate_utils import *

keywords_list = []
# kw_model = KeyBERT(model="distiluse-base-multilingual-cased-v2")
kw_model = KeyBERT()
kw_model_viet = KeyBERT(model="distiluse-base-multilingual-cased-v2")

rootdir = "viet_articles"
for file in os.listdir(rootdir):
    with open(os.path.join(rootdir, file), 'r') as f:
        content = json.load(f)
        title = content['title']
        lang = content['language']

        if lang == "en":
        # print(title)
            keywords = kw_model.extract_keywords(title, stop_words="english", keyphrase_ngram_range=(1, 1), top_n=5)
            sorted_keywords = sorted(keywords, key=lambda tup: tup[1], reverse=True)
            # print(sorted_keywords)
            selected_keywords = [i[0] for i in keywords]

            content["keywords"] = selected_keywords
            path_to_save = rootdir+"_with_keywords"
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)

            with open(os.path.join(path_to_save, file), 'w') as handle:
                json.dump(content, handle, indent=4)

        elif lang == "vi":
            translated_title = translate_text_with_source(source="vi", target="en", text=title)
            content["title_eng"] = translated_title
            # print(title)
            # print(translated_title)
            keywords_viet = kw_model_viet.extract_keywords(title, keyphrase_ngram_range=(1, 1), top_n=5)
            print(keywords_viet)
            selected_keywords = [i[0] for i in keywords_viet]
            content["keywords"] = selected_keywords
            keywords_eng = [translate_text_with_source(source="vi", target="en", text=word) for word in selected_keywords]
            content["keywords_eng"] = keywords_eng
            # keywords = kw_model.extract_keywords(translated_title, stop_words="english", keyphrase_ngram_range=(1, 1), top_n=5)
            # sorted_keywords = sorted(keywords, key=lambda tup: tup[1], reverse=True)
            # # print(sorted_keywords)
            # selected_keywords = [i[0] for i in keywords]
            # content["keywords_eng"] = selected_keywords
            # keywords_viet = [translate_text_with_source(source="en", target="vi", text=word) for word in selected_keywords]
            # content["keywords"] = keywords_viet

            path_to_save = rootdir + "_with_keywords"
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

