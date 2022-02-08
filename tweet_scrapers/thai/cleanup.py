import os
import json
import re
import pandas as pd
from dateutil import parser
import ast
from thai_dateutils import thai_date_convert

titles = []
content = []

df = pd.read_csv("Thai_Antifakenewscenter_csvdump.csv")
print(df.head())

for index, row in df.iterrows():
    link = ast.literal_eval(row['link'])[0]
    # print(link)
    title = row['Title']
    print(title)
    tags = ast.literal_eval(row["Tags"])

    '''
    There are two articles that are general introductions
    They have neither language tag in them
    These 2 will be discarded
    '''

    # # print(tags)
    # if "Tiếng Việt" in tags:
    #     lang = "vi"
    # elif "English" in tags:
    #     lang = "en"
    # else:
    #     lang = "discard"
    lang = "th"

    paras = ast.literal_eval(row["Paragraphs"])[1:] #first of paragraphs is just a language tag
    images = ast.literal_eval(row['Image Sources'])
    content_links = ast.literal_eval(row["Links in content"])
    author = row["Author"]
    date = thai_date_convert(row["date"].replace('\r', '').replace('\n', '').strip()).isoformat()
    # print(date)
    veracity = row["Truefalse"]
    category = row["Category"]

    if lang != "discard":
        content_dict = {"title": title,
                        "tags": tags,
                        "paragraphs": paras,
                        "images": images,
                        "content_links": content_links,
                        "author": author,
                        "date": date,
                        "veracity": veracity,
                        "category": category,
                        "language": lang}
        with open(os.path.join("thai_articles", "{}.json".format(str(index).zfill(4))), "w") as handle:
            json.dump(content_dict, handle, indent=4)


# for file in os.listdir(rootdir):
#     with open(os.path.join(rootdir, file), 'r') as f:
#         content = json.load(f)
#
#     title = content['Title']
#     title_clean = re.sub(r"[\n\t\[\]]*", "", title)
#     content["Title"] = title_clean
#     # print(title_clean)
#
#     # para = content["Paragraphs"]
#     # link = content['link']
#     # veracity = content["Truefalse"]
#     #
#     # cleaned_content = {"title": title_clean,
#     #                    "paragraphs": para,
#     #                    "link": link,
#     #                    "veracity": veracity}
#
#     with open(os.path.join("blackdot_cleaned", file), "w") as handle:
#         json.dump(content, handle, indent=4)
