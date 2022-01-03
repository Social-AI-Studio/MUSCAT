import os
import json
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download("stopwords")
# nltk.download("punkt")

label_connotation = {"unverified":0,"non-rumor":1,"true":2,"false":3,0:"unverified",1:"non-rumor",2:"true",3:"false"}
stop_words = set(stopwords.words('english'))

filetargets = [("twitter15","twitter15BU","twitter15TD"),("twitter16","twitter16BU","twitter16TD")]
sourcedict = {}
labeldict = {}
alltokens = {}
tokencounter = 0
tokencounts = {}
for i in ["twitter15","twitter16"]:
    with open(os.path.join(i,"source_tweets.txt"),"r",encoding="utf-8") as tweet_text_file:
        for line in tweet_text_file:
            if line:
                # print(repr(line))
                linelist = line.strip().split("\t")
                tokens = word_tokenize("\t".join(linelist[1:]))
                # filtered = [w for w in tokens if not w.lower() in stop_words]
                filtered = tokens
                for newcandidate in filtered:
                    if not newcandidate in alltokens:
                        alltokens[newcandidate] = tokencounter
                        tokencounts[newcandidate] = 0
                        tokencounter+=1
                        tokencounts[newcandidate] = tokencounts[newcandidate] + 1

                sourcedict[int(linelist[0])] = ["\t".join(linelist[1:]),filtered]
                
    with open(os.path.join(i,"label.txt"),"r",encoding="utf-8") as labelfile:
        for line in labelfile:
            if line:
                # print(repr(line))
                linelist = line.strip().split(":")
                labeldict[linelist[1]] = label_connotation[linelist[0]]

print("Before:",len(list(sourcedict.keys())))
with open("pulled_tweets.json","r",encoding="utf-8") as pulled_tweets:
    items = json.load(pulled_tweets)
    for tweet_id in items:
        if items[tweet_id]!=None:
            tokens = word_tokenize(items[tweet_id])
            # filtered = [w for w in tokens if not w.lower() in stop_words]
            filtered = tokens
            for newcandidate in filtered:
                if not newcandidate in alltokens:
                    alltokens[newcandidate] = tokencounter
                    tokencounts[newcandidate] = 0
                    tokencounter+=1
                    tokencounts[newcandidate] = tokencounts[newcandidate] + 1
            sourcedict[int(tweet_id)] = [items[tweet_id],filtered]
            
print("After:",len(list(sourcedict.keys())))
tokencounts = {a: b for a,b in sorted(tokencounts.items(),key=lambda item:item[1], reverse=True)}
BUValids = set(list(tokencounts.keys())[:3229]) # BU has 3229 unique vocabs in her counts..
TDValids = set(list(tokencounts.keys())[:3260]) # TD has 3260 unique vocabs in her counts..

# IGNORE MA'S LIMIT.
BUValids = set(list(tokencounts.keys()))
TDValids = set(list(tokencounts.keys()))
tdvocabdict = {}
counter = 0
for i in TDValids:
    tdvocabdict[i] = counter
    counter+=1
buvocabdict = {}
counter = 0
for i in BUValids:
    buvocabdict[i] = counter
    counter+=1



# print(len(list(tokencounts.keys()))) # 4954 AFTER stopword removal just on twitter 15 alone... 6167 for both btw.
# doesn't quite match with her number of unique vocabs. so we CUT it based off number of occurrences.


for tweet_id in sourcedict: # perform further censoring based off top vocab used based off her total vocab count.
    tokenlist = sourcedict[tweet_id][1]
    TDver = {}
    BUver = {}
    for tokenfocus in tokenlist:
        if tokenfocus in TDValids:
            if not tokenfocus in TDver:
                TDver[tokenfocus] = 0 
            TDver[tokenfocus] = TDver[tokenfocus] + 1
        if tokenfocus in BUValids:
            if not tokenfocus in BUver:
                BUver[tokenfocus] = 0 
            BUver[tokenfocus] = BUver[tokenfocus] + 1
    sourcedict[tweet_id] = [sourcedict[tweet_id][0],sourcedict[tweet_id][1],TDver,BUver]
        # original, filtered, TDcount, BU count

# print(sourcedict[list(sourcedict.keys())[0]])
# vectorizer = TfidfVectorizer()
# vectorizer.fit_transform(alltexts)
# print(vectorizer.get_feature_names_out())

# for ss in list(sourcedict.keys()):
    # target = sourcedict[ss]
    # z = vectorizer.transform(target[1])
    # print(i)
    # print("Tweetname:",ss)
    # print(type(z.todok()))
    # print(z.todok())
    # print(z.todok().keys())
    # input()
# quit()



currentcount = 0
all_BU = []
all_TD = []
for dataset in ["twitter15","twitter16"]:
    for treefile in os.listdir(os.path.join(dataset,"tree")):
        root = treefile.replace(".txt","")
        recordstr_BU = 0
        recordstr_TD = 0
        parentalcount_td = set()
        parentalcount_bu = set()
        held_BU = []
        held_TD = []
        helditems = {}

        with open(os.path.join(dataset,"tree",treefile)) as openedfile:
            for line in openedfile:
                if line:
                    splits = line.split("->")
                    parent = ast.literal_eval(splits[0])
                    child = ast.literal_eval(splits[1])
                    if parent[0]=="ROOT":
                        parent[1] = root
                    # 1: root-id -- an unique identifier describing the tree (tweetid of the root);
                    # 2: index-of-parent-tweet -- an index number of the parent tweet for the current tweet;
                    # 3: index-of-the-current-tweet -- an index number of the current tweet;
                    # 4: parent-number -- the total number of the parent node in the tree that the current tweet is belong to;
                    # 5: text-length -- the maximum length of all the texts from the tree that the current tweet is belong to;
                    # 6: vocab counts. (she doesn't mention this in documentation, but it's supposedly this.)
                    if not int(parent[1]) in sourcedict:
                        continue
                    if not int(child[1]) in sourcedict: # BANDAID TO IGNORE TWEETS MISSING TEXT.
                        continue




                    
                    if not parent[1] in helditems:
                        helditems[parent[1]] = [parent[1],set(),set(),currentcount,None,None] 
                        # id, tdidxlistparent,buidxlistparent, tweetidx,numberofparents,textlength
                        
                        # instead of appending all parents to a list, Ma just ignores retweets.
                        currentcount+=1
                    parentalcount_td.add(parent[1])
                    helditems[parent[1]][2].add(child[1]) #add to set of parents (BU)
                   
                    if not child[1] in helditems:
                        helditems[child[1]] = [child[1],set(),set(),currentcount,None,None]
                        currentcount+=1
                    parentalcount_bu.add(child[1]) # we need to count the number of unique parents.
                    helditems[child[1]][1].add(parent[1]) # add to set of parents (TD)
                    
                    # ['ROOT', 'ROOT', '0.0']->['972651', '80080680482123777', '0.0']        
                    # ['uid', 'tweet ID', 'post time delay (in minutes)']
        # print(os.path.join(dataset,"tree",treefile))
        # print(len(list(helditems.keys())))
        for tweet_instance in helditems:
            sourcelist = helditems[tweet_instance]  
            td = sourcedict[int(tweet_instance)][2] # the respective vocab count versions are pulled here.
            bu = sourcedict[int(tweet_instance)][3]
            
            if len(list(td.keys()))>recordstr_TD: # search for the max vocab counts used in the tweets for this tree.
                recordstr_TD = len(list(td.keys()))
            if len(list(bu.keys()))> recordstr_BU:
                recordstr_BU = len(list(bu.keys()))
        for tweet_instance in helditems:
            sourcelist = helditems[tweet_instance]
            td = sourcedict[int(tweet_instance)][2]
            bu = sourcedict[int(tweet_instance)][3] # again, pull the respective vocab count versions.
            # print(helditems[tweet_instance][3])
            tdver = [root,list(sourcelist[1]),helditems[tweet_instance][3],len(list(parentalcount_td)),recordstr_TD,td]
            if helditems[tweet_instance][0]==root: # tree root always is none
                tdver[1]="None"
            else:
                if len(sourcelist[1])==1:
                    tdver[1] = helditems[list(sourcelist[1])[0]][3]
                else:
                    if not list(sourcelist[1]):
                        # print(helditems[tweet_instance])
                        # print(root)
                        # input()
                        tdver[1] = "None" # So in this case there is no parent.. but it also wasn't a source node. what is it?
                        # the answer is that it is a node that has a parent that does not have the tweet text available.
                        tdver = []
                        # mark for deletion.
                    else:
                        tdver[1] = helditems[list(sourcelist[1])[-1]][3] # we just chunk the last parent as the main parent.
                    # print("tdver: ERROR") # should not be triggered. Just here in case, but probably will be.
                    # print(tdver)
                    # input()
                    
            buver = [root,list(sourcelist[2]),helditems[tweet_instance][3],len(list(parentalcount_bu)),recordstr_BU,bu]
            if helditems[tweet_instance][0]==root : # tree root always is none.. for some reason. even in BU
                buver[1]="None"
            else:
                if len(sourcelist[2])==1:
                    buver[1] = helditems[list(sourcelist[2])[0]][3]
                else: # more than one parent, as is should be, but not explained in their paper either.
                    if not list(sourcelist[2]):
                        buver[1] = "None" # So in this case there is no parent if you're talking bottom up. this is an absolute leaf on the tree.
                        # Ma just.. regularly trees it but... it's not correct. So if there is no parent, i push None as an item.
                    else:
                        buver[1] = helditems[list(sourcelist[2])[-1]][3] # we just chunk the last parent as the main parent... despite it having several parents.
                    
                    
                    
                    # she doesn't account for this in her preprocessing either because multiple parents for bottom up should be the norm, yet nothing is done to account for this in code.
                    # print("buver") # Should be triggered all the time basically.
                    # print(buver)
                    # input()
            
            # 1: root-id -- an unique identifier describing the tree (tweetid of the root);
            # 2: index-of-parent-tweet -- an index number of the parent tweet for the current tweet;
            # 3: index-of-the-current-tweet -- an index number of the current tweet;
            # 4: parent-number -- the total number of the parent node in the tree that the current tweet is belong to;
            # 5: text-length -- the maximum length of all the texts from the tree that the current tweet is belong to;
            # 6: vocab counts. (she doesn't mention this in documentation, but it's supposedly this.)
            # to reiterate, this is Ma's Format
            
            # note that Ma's format is also inconsistent in the index numbers. Sometimes she refreshes between trees, sometimes she doesn't.
            # Ma's format also adds or removes some tweets randomly into trees later on. (tree.txt doesn't have the 
            if tdver:
                held_TD.append(tdver) # ma also counts no parents as a parent.. therefore we should add 1 to all.
            if buver:
                held_BU.append(buver)
        all_TD.extend(held_TD)
        all_BU.extend(held_BU)

with open("shaun_TD.txt","w",encoding="utf-8") as tdfile:
    for i in all_TD:
        additional = []
        for item in i[-1]:
            additional.append(str(tdvocabdict[item]) + ":" + str(i[-1][item]))
        i.pop()
        i.extend(additional)
        appender = []
        for item in i:
            if not type(item)==type([]):
                appender.append(str(item))
            elif len(item)==1:
                appender.append(str(item[0]))
                # print(i)
            else:
                print(i)
                
        # print(appender)
        tdfile.write("\t".join(appender))
        tdfile.write("\n")
        
with open("shaun_BU.txt","w",encoding="utf-8") as bufile:
    for i in all_BU:
        additional = []
        for item in i[-1]:
            additional.append(str(buvocabdict[item]) + ":" + str(i[-1][item]))
        i.pop()
        i.extend(additional)
        appender = []
        for item in i:
            if not type(item)==type([]):
                appender.append(str(item))
            elif len(item)==1:
                appender.append(str(item[0]))
                # print(i)
            else:
                print(i)
        bufile.write("\t".join(appender))
        bufile.write("\n")
