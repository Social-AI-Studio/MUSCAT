import os
import json
import re
import pandas as pd

titles = []
content = []

rootdir = "blackdotdump"

for file in os.listdir(rootdir):
    with open(os.path.join(rootdir, file), 'r') as f:
        content = json.load(f)

    title = content['Title']
    title_clean = re.sub(r"[\n\t\[\]]*", "", title)
    content["Title"] = title_clean
    # print(title_clean)

    # para = content["Paragraphs"]
    # link = content['link']
    # veracity = content["Truefalse"]
    #
    # cleaned_content = {"title": title_clean,
    #                    "paragraphs": para,
    #                    "link": link,
    #                    "veracity": veracity}

    with open(os.path.join("blackdot_cleaned", file), "w") as handle:
        json.dump(content, handle, indent=4)
