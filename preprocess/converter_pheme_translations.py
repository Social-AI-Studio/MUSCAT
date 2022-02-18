import logging
import os
import json
import ast
import logging
from tqdm import tqdm
from utils import preprocess_en_text, Vocabulary
from rand5fold import load5foldData
from convert_veracity_annotations import convert_annotations
from nltk.tokenize import TweetTokenizer


logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s:%(lineno)d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_VOCAB_SZ = 5000
t_vocab = Vocabulary(max_vocab_size=MAX_VOCAB_SZ)
w_tokenizer = TweetTokenizer()


def get_tweet_text(filename: str):
    with open(filename, "r") as f:
        data = json.load(f)
        logger.debug(data.get("translated_text"))
    return data.get("translated_text")


def vocab_gen():

    dir_name = "pheme-translations/all-rnr-annotated-threads/"
    sub_dirs = os.listdir(dir_name)

    cntr = 0
    for event_dir in sub_dirs:
        if event_dir.startswith("."):
            continue
        logger.info("\t-------------------------------------")
        logger.info("Processing event {}".format(event_dir))
        files = os.listdir(os.path.join(dir_name, event_dir))
        for file_name in files:
            if file_name.startswith("."):
                continue

            logger.info("***** Working on {} dir *****".format(file_name))
            annot_subfile = os.path.join(dir_name, event_dir, file_name)
            tweet_tree_dirs = os.listdir(annot_subfile)
            cntr += len(tweet_tree_dirs)
            logger.debug("{} {}".format(len(tweet_tree_dirs), cntr))
            for idx, tweet_tree_idx in enumerate(tqdm(tweet_tree_dirs)):
                if tweet_tree_idx.startswith("."):
                    continue
                logger.debug(
                    "***** Processing tweet id {} *****".format(tweet_tree_idx)
                )
                # get root and reaction tweet text
                t_proc = TreeProcessor(annot_subfile, tweet_tree_idx, lang="id")
                root_tweet = t_proc.get_source_tweet()
                t_vocab.add_sentence_to_corpus(root_tweet)

                if not os.path.isdir(t_proc.reaction_tweet_dir):
                    continue
                for reation_idx in os.listdir(t_proc.reaction_tweet_dir):
                    if reation_idx.startswith("."):
                        continue
                    cur_reaction_tweet = t_proc.get_reaction_tweet(reation_idx)
                    logger.debug(f"reaction --> {cur_reaction_tweet}")
                    t_vocab.add_sentence_to_corpus(cur_reaction_tweet)
    t_vocab.reset_vocab()
    logger.info(f"total sentences --> {t_vocab.num_sentences}")


def reader_pheme(lang: str):
    # generating vocabulary
    vocab_gen()

    lines_to_dump = []
    labels_to_dump = []
    dir_name = "pheme-translations/all-rnr-annotated-threads/"
    sub_dirs = os.listdir(dir_name)

    cntr = 0
    for event_dir in sub_dirs:
        if event_dir.startswith("."):
            continue
        logger.info("\t-------------------------------------")
        logger.info("Processing event {}".format(event_dir))
        files = os.listdir(os.path.join(dir_name, event_dir))

        for file_name in files:
            if file_name.startswith("."):
                continue

            logger.info("***** Working on {} dir *****".format(file_name))
            annot_subfile = os.path.join(dir_name, event_dir, file_name)
            tweet_tree_dirs = os.listdir(annot_subfile)
            cntr += len(tweet_tree_dirs)
            logger.debug("{} {}".format(len(tweet_tree_dirs), cntr))
            for idx, tweet_tree_idx in enumerate(tqdm(tweet_tree_dirs)):
                if tweet_tree_idx.startswith("."):
                    continue
                logger.debug(
                    "***** Processing tweet id {} *****".format(tweet_tree_idx)
                )
                # get root tweet text, tree structure
                t_proc = TreeProcessor(annot_subfile, tweet_tree_idx, lang)
                edge_list, num_par, maxlen = t_proc.get_tree_structure()

                # ignore trees that has no childrens
                if edge_list:
                    # get tree label
                    label = t_proc.get_label()
                    if label is None:
                        label = "non-rumor"

                    logger.debug(f"num parents --> {num_par}")
                    logger.info(f"num edge_list --> {edge_list}")

                    labels_to_dump.append(f"{label}\tEMPTY\t{tweet_tree_idx}")
                    logger.debug(f"label --> {label}")

                    tree_lns = []
                    for item in edge_list:
                        cur_ln = ""
                        for i in item:
                            cur_ln += str(i) + "\t"
                        tree_lns.append(cur_ln)

                    lines_to_dump.extend(tree_lns)

                if (idx + 1) % 500 == 0:
                    logger.info("*** Processed {} tweets ***".format(idx + 1))

    out_dir = f"preprocess/PHEME/{lang.upper()}"
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"data.TD_RvNN.vol_{MAX_VOCAB_SZ}.txt")
    f = open(outfile, "w")
    f.writelines("\n".join(lines_to_dump))
    f.close()

    outfile = os.path.join(out_dir, f"PHEME_label_All_ID.txt")
    f = open(outfile, "w")
    f.writelines("\n".join(labels_to_dump))
    f.close()

    load5foldData(obj="PHEME")


class TreeProcessor:
    def __init__(self, annot_subfile: str, tweet_tree_idx: str, lang: str):
        self.lang = lang
        self.tweet_tree_idx = tweet_tree_idx
        self.tweet_tree_path = os.path.join(annot_subfile, tweet_tree_idx)
        self.reaction_tweet_dir = os.path.join(
            self.tweet_tree_path, "reactions", self.lang
        )
        self.tweet_tree_p = "/".join(self.tweet_tree_path.split("/")[1:])
        self.root_tweet_structure_path = os.path.join(
            self.tweet_tree_p, "structure.json"
        )
        self.tweet_indices_in_tree = {}
        self.node_indices = 0

    def recurse_tree(self, data, num_par, maxlen, edge_list, root=None):
        if not data:
            return num_par, maxlen, edge_list

        num_par += 1
        for children in data.keys():
            if root:
                cur_tweet = self.get_reaction_tweet(children)
            else:
                cur_tweet = self.get_source_tweet()
            cur_tweet = preprocess_en_text(cur_tweet)
            cur_tokens = w_tokenizer.tokenize(cur_tweet)
            index_cnt_list, cur_len = t_vocab.get_vocab_count(cur_tokens)

            if cur_len < 1:
                continue

            maxlen = max(maxlen, cur_len)
            if cur_tweet:
                # create index for children
                if self.tweet_indices_in_tree.get(children) is None:
                    self.node_indices = self.node_indices + 1
                    self.tweet_indices_in_tree[children] = self.node_indices
                logger.debug(f"{root}-->{children}")
                edge_list.append(
                    (
                        self.tweet_tree_idx,
                        self.tweet_indices_in_tree.get(root),
                        self.tweet_indices_in_tree[children],
                        index_cnt_list,
                    )
                )

            num_par, maxlen, edge_list = self.recurse_tree(
                data[children], num_par, maxlen, edge_list, children
            )

        return num_par, maxlen, edge_list

    def get_source_tweet(self):
        root_tweet_filepath = (
            os.path.join(
                self.tweet_tree_path, self.lang, "source-tweets", self.tweet_tree_idx
            )
            + ".json"
        )
        logger.debug(root_tweet_filepath)
        root_tweet = get_tweet_text(root_tweet_filepath)
        return root_tweet

    def get_reaction_tweet(self, idx):
        reaction_file = idx + ".json" if not idx.endswith(".json") else idx
        logger.debug(f"reaction path --> {reaction_file}")
        reaction_path = os.path.join(self.reaction_tweet_dir, reaction_file)
        cur_reaction_tweet = ""

        try:
            cur_reaction_tweet = get_tweet_text(reaction_path)
            logger.debug(f"reaction tweet --> {cur_reaction_tweet}")
        except Exception as e:
            pass

        return cur_reaction_tweet

    def get_label(self):
        root_tweet_annot_path = os.path.join(self.tweet_tree_p, "annotation.json")
        with open(root_tweet_annot_path, "r") as f:
            data = json.load(f)
        label = convert_annotations(data)
        return label

    def get_tree_structure(self):
        with open(self.root_tweet_structure_path, "r") as f:
            data = json.load(f)
        logger.debug(f"data --> {data}")
        num_par, maxlen, edge_list = self.recurse_tree(
            data, num_par=0, maxlen=0, edge_list=[]
        )

        # ignoring tree with only root node
        if num_par == 1:
            return None, None, None

        updated_edge_list = []
        for cur_edge in edge_list:
            updated_edge = (
                cur_edge[:3]
                + (
                    num_par,
                    maxlen,
                )
                + (cur_edge[3],)
            )
            updated_edge_list.append(updated_edge)

        return updated_edge_list, num_par, maxlen


if __name__ == "__main__":
    lang = "id"
    reader_pheme(lang)
    # vocab_gen()
