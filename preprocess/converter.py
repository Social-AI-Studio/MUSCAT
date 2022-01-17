# preprocess file that converts raw tweets
import logging
import os
import json
import ast
import logging
from tqdm import tqdm
from utils import preprocess_en_text, get_tfidf_top_features, Vocabulary
from nltk.tokenize import TweetTokenizer


logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s:%(lineno)d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_VOCAB_SZ = 5000
w_tokenizer = TweetTokenizer()


def reader_twitter15_16():
    t_vocab = Vocabulary(max_vocab_size=MAX_VOCAB_SZ)
    tweet_id_mapped = {}
    root_dir = "rumor_detection_acl2017"
    sub_dirs = ["twitter15", "twitter16"]
    for sub_dir in sub_dirs:
        file_name = os.path.join(root_dir, sub_dir, "source_tweets.txt")
        num_lines = sum(1 for line in open(file_name, "r"))
        with open(file_name, "r") as f:
            for i, line in enumerate(tqdm(f, total=num_lines)):
                line = line.strip()
                splits = line.split("\t")
                tid = splits[0]
                tweet = splits[1]
                # print(f"id {tid}, tweet {tweet}")
                t_vocab.add_sentence_to_corpus(tweet)
                tweet_id_mapped[tid] = tweet

    with open("RvNN-pytorch/preprocessing/pulled_tweets.json", "r") as f:
        items = json.load(f)
        for tweet_id in tqdm(items):
            if items[tweet_id] != None:
                t_vocab.add_sentence_to_corpus(items[tweet_id])
                tweet_id_mapped[tweet_id] = items[tweet_id]
    t_vocab.reset_vocab()
    logger.info(t_vocab.word2index)
    return tweet_id_mapped, t_vocab


"""
1: root-id -- an unique identifier describing the tree (tweetid of the root);

2: index-of-parent-tweet -- an index number of the parent tweet for the current tweet;

3: index-of-the-current-tweet -- an index number of the current tweet;

4: parent-number -- the total number of the parent node in the tree that the current tweet is belong to;

5: text-length -- the maximum length of all the texts from the tree that the current tweet is belong to;

6: list-of-index-and-counts -- the rest of the line contains space separated index-count pairs, where a index-count pair is in format of "index:count", E.g., "index1:count1 index2:count2" (extracted from the "text" field in the json format from Twitter)

"""


def process_trees_to_network_input():
    tweet_id_mapped, t_vocab = reader_twitter15_16()
    root_dir = "rumor_detection_acl2017"
    sub_dirs = ["twitter15", "twitter16"]
    lines_to_dump = []

    for sub_dir in sub_dirs:
        tree_dir = os.path.join(root_dir, sub_dir, "tree")
        tree_files = os.listdir(tree_dir)
        for fid, file_name in enumerate(tree_files):
            # print(f"processing ... {file_name}")
            root = file_name.strip()[:-4]
            is_root = False
            node_indices = 0
            parent_node_set = set()
            childrens = []
            max_tweet_len = 0
            tree = []
            tweet_indices_in_tree = {}
            file_name = os.path.join(tree_dir, file_name)
            with open(file_name, "r") as f:
                for idx, line in enumerate(f):
                    splits = line.split("->")
                    parent = ast.literal_eval(splits[0])
                    children = ast.literal_eval(splits[1])

                    p_uid = parent[0]
                    p_id = parent[1]
                    p_time = parent[2]
                    # print(f"p_uid {p_uid}, p_id {p_id} p_time {p_time}")

                    ch_uid = children[0]
                    ch_id = children[1]
                    ch_time = children[2]
                    # print(f"ch_uid {ch_uid}, ch_id {ch_id} ch_time {ch_time}")

                    # checking if tweet has tokens inside vocab limit, otherwise it creates empty line
                    fflag = True
                    cur_tokens = None
                    if tweet_id_mapped.get(ch_id):
                        cur_tweet = preprocess_en_text(tweet_id_mapped[ch_id])
                        cur_tokens = w_tokenizer.tokenize(cur_tweet)
                        for word in cur_tokens:
                            if t_vocab.word2index.get(word) is not None:
                                fflag = False
                    if fflag:
                        continue

                    # check if tweet source exits, validating children and ignoring retweets
                    if (
                        p_id == ch_id
                        or tweet_id_mapped.get(ch_id) is None
                        or tweet_indices_in_tree.get(ch_id) in childrens
                    ):
                        continue

                    # validating root node
                    if p_id != "ROOT" and tweet_indices_in_tree.get(p_id) is None:
                        continue

                    # create index for children
                    if tweet_indices_in_tree.get(ch_id) is None:
                        node_indices = node_indices + 1
                        tweet_indices_in_tree[ch_id] = node_indices

                    if (
                        p_id != "ROOT"
                        and tweet_indices_in_tree[p_id] not in parent_node_set
                    ):
                        parent_node_set.add(tweet_indices_in_tree[p_id])

                    index_cnt_list, cur_len = t_vocab.get_vocab_count(cur_tokens)
                    max_tweet_len = max(max_tweet_len, cur_len)
                    println = f"{root}\t{tweet_indices_in_tree.get(p_id)}\t{tweet_indices_in_tree[ch_id]}\t{index_cnt_list}"
                    tree.append(
                        [
                            root,
                            None if p_id == "ROOT" else tweet_indices_in_tree.get(p_id),
                            tweet_indices_in_tree.get(ch_id),
                            None,
                            None,
                            index_cnt_list,
                        ]
                    )
                    childrens.append(tweet_indices_in_tree[ch_id])
            tree_lns = []
            for item in tree:
                item[3] = len(parent_node_set)
                item[4] = max_tweet_len
                cur_ln = ""
                for i in item:
                    cur_ln += str(i) + "\t"
                tree_lns.append(cur_ln)

                # confirming tree has at least one node that has no parent
                if item[1] == None:
                    is_root = True

            if is_root:
                lines_to_dump.extend(tree_lns)

            logger.info(f"total lines to dump {len(lines_to_dump)}")

            # if fid > 5:
            #     break
    out_dir = "preprocess"
    outfile = os.path.join(out_dir, f"twtiter15_voc{MAX_VOCAB_SZ}.txt")
    f = open(outfile, "w")
    f.writelines("\n".join(lines_to_dump))
    f.close()


if __name__ == "__main__":
    process_trees_to_network_input()
    # reader_twitter15_16()
