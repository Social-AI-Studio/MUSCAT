import sys
import re
import os
import json
import logging
import time
import json
from tqdm import tqdm

lib_path = os.path.abspath(os.path.join(__file__, "..", "..", "preprocess"))
sys.path.append(lib_path)
from convert_veracity_annotations import convert_annotations

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s:%(lineno)d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

par_dir = os.path.dirname(os.getcwd())

event_whitelist = [
    "charliehebdo-all-rnr-threads",
    "sydneysiege-all-rnr-threads",
    "ottawashooting-all-rnr-threads",
    "ferguson-all-rnr-threads",
    "germanwings-crash-all-rnr-threads",
]


def convert_pheme_sequential(lang: str):
    dir_name = os.path.join(par_dir, "pheme-translations/all-rnr-annotated-threads")
    sub_dirs = os.listdir(dir_name)

    pheme_info_all = {}
    label_cntr = {}
    pbar = tqdm(range(len(sub_dirs)), desc="Event")
    for event_dir in sub_dirs:
        if event_dir.startswith("."):
            continue
        logger.debug("\t-------------------------------------")
        logger.debug("Processing event {}".format(event_dir))
        files = os.listdir(os.path.join(dir_name, event_dir))
        pheme_info_all[event_dir] = []
        for file_name in files:
            if file_name.startswith("."):
                continue
            logger.debug("***** Working on {} dir *****".format(file_name))
            annot_subfile = os.path.join(dir_name, event_dir, file_name)
            tweet_tree_dirs = os.listdir(annot_subfile)
            for idx, tweet_tree_idx in enumerate(tweet_tree_dirs):
                pbar.set_postfix({"Tree": idx})
                time.sleep(0.0001)
                if tweet_tree_idx.startswith("."):
                    continue
                logger.debug(
                    "***** Processing tweet id {} *****".format(tweet_tree_idx)
                )
                data_item_proc = DataProcessor(annot_subfile, tweet_tree_idx, lang)
                tweet_thread = data_item_proc.get_thread_sequential()
                label = data_item_proc.get_label()
                if label is None:
                    label = 3

                if label_cntr.get(str(label)) is None:
                    label_cntr[str(label)] = 0
                label_cntr[str(label)] += 1

                pheme_info_all[event_dir].append(
                    json.dumps(
                        {
                            "id_": tweet_tree_idx,
                            "label": label,
                            "tweets": tweet_thread,
                        }
                    )
                )
                logger.debug(
                    f"id_: {tweet_tree_idx}, label: {label}, tweets: {tweet_thread}"
                )

    logger.info(label_cntr)
    outdir = os.path.join("rumor_data/pheme4cls", lang)
    os.makedirs(outdir)
    for i, split_key in enumerate(event_whitelist):
        test_identifiers = pheme_info_all[split_key]
        train_identifiers = []

        for it_key in pheme_info_all:
            if it_key != split_key:
                train_identifiers.extend(pheme_info_all[it_key])
        logger.info(
            f"Loading fold {i}# {split_key} \t sample size: train {len(train_identifiers)}, test {len(test_identifiers)}"
        )
        fold = "-".join(split_key.split("-")[:-3])
        foutdir = os.path.join(outdir, f"{fold}")
        os.makedirs(foutdir, exist_ok=True)
        train_fname = os.path.join(foutdir, "train.json")
        test_fname = os.path.join(foutdir, "test.json")
        with open(train_fname, "w") as filehandle:
            filehandle.writelines(
                "%s\n" % tree_idx for tree_idx in list(train_identifiers)
            )

        with open(test_fname, "w") as filehandle:
            filehandle.writelines(
                "%s\n" % tree_idx for tree_idx in list(test_identifiers)
            )


class DataProcessor:
    def __init__(self, annot_subfile: str, tweet_tree_idx: str, lang: str):
        self.lang = lang
        self.tweet_tree_idx = tweet_tree_idx
        self.tweet_tree_path = os.path.join(annot_subfile, tweet_tree_idx)
        self.reaction_tweet_dir = os.path.join(self.tweet_tree_path, "reactions", lang)
        self.tweet_tree_p = self.tweet_tree_path.replace("pheme-translations/", "")
        logger.info(self.tweet_tree_p)
        self.root_tweet_structure_path = os.path.join(
            self.tweet_tree_p, "structure.json"
        )
        self.tweet_indices_in_tree = {}
        self.node_indices = 0

    def get_tweet_text(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)
            logger.debug(data.get("translated_text"))
        return data.get("translated_text")

    def get_source_tweet(self):
        root_tweet_filepath = (
            os.path.join(
                self.tweet_tree_path, self.lang, "source-tweets", self.tweet_tree_idx
            )
            + ".json"
        )
        root_tweet = self.get_tweet_text(root_tweet_filepath)
        return root_tweet

    def recurse_tree(self, data, edge_list, root=None):
        if not data:
            return edge_list

        for children in data.keys():
            if root:
                cur_tweet = self.get_reaction_tweet(children)
            else:
                cur_tweet = self.get_source_tweet()
            cur_tweet = re.sub("\n|\r", " ", cur_tweet)
            if cur_tweet:
                # create index for children
                if self.tweet_indices_in_tree.get(children) is None:
                    self.tweet_indices_in_tree[children] = self.node_indices
                logger.debug(f"{root}-->{children}")
                edge_list.append(cur_tweet)

            edge_list = self.recurse_tree(data[children], edge_list, children)

        return edge_list

    def get_reaction_tweet(self, idx):
        reaction_file = idx + ".json" if not idx.endswith(".json") else idx
        logger.debug(f"reaction path --> {reaction_file}")
        reaction_path = os.path.join(self.reaction_tweet_dir, reaction_file)
        cur_reaction_tweet = ""

        try:
            cur_reaction_tweet = self.get_tweet_text(reaction_path)
            logger.debug(f"reaction tweet --> {cur_reaction_tweet}")
        except Exception as e:
            pass

        return cur_reaction_tweet

    def get_label(self):
        root_tweet_annot_path = os.path.join(self.tweet_tree_p, "annotation.json")
        with open(root_tweet_annot_path, "r") as f:
            data = json.load(f)
        label = convert_annotations(data, string=False)
        return label

    def get_thread_sequential(self):
        with open(self.root_tweet_structure_path, "r") as f:
            data = json.load(f)
        logger.debug(f"data --> {data}")
        tweet_concatenated_list = self.recurse_tree(data, edge_list=[])
        return tweet_concatenated_list


if __name__ == "__main__":
    lang = sys.argv[1]
    convert_pheme_sequential(lang)
