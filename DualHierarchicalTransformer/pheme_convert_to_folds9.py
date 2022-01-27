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


def convert_pheme_sequential():
    dir_name = os.path.join(par_dir, "all-rnr-annotated-threads")
    sub_dirs = os.listdir(dir_name)

    pheme_info_all = {}
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
                data_item_proc = DataProcessor(annot_subfile, tweet_tree_idx)
                tweet_thread = data_item_proc.get_thread_sequential()
                label = data_item_proc.get_label()
                if not label:
                    label = 3

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
    outdir = "rumor_data/pheme4cls"
    for i, split_key in enumerate(pheme_info_all):
        test_identifiers = pheme_info_all[split_key]
        train_identifiers = []

        for it_key in pheme_info_all:
            if it_key != split_key:
                train_identifiers.extend(pheme_info_all[it_key])
        logger.info(
            f"Loading fold {i}# {split_key} \t sample size: train {len(train_identifiers)}, test {len(test_identifiers)}"
        )
        foutdir = os.path.join(outdir, f"split_{i}")
        os.makedirs(foutdir, exist_ok=True)
        test_fname = os.path.join(foutdir, "train.json")
        train_fname = os.path.join(foutdir, "test.json")
        with open(train_fname, "w") as filehandle:
            filehandle.writelines(
                "%s\n" % tree_idx for tree_idx in list(train_identifiers)
            )

        with open(test_fname, "w") as filehandle:
            filehandle.writelines(
                "%s\n" % tree_idx for tree_idx in list(test_identifiers)
            )


class DataProcessor:
    def __init__(self, annot_subfile: str, tweet_tree_idx: str):
        self.tweet_tree_idx = tweet_tree_idx
        self.tweet_tree_path = os.path.join(annot_subfile, tweet_tree_idx)
        self.reaction_tweet_dir = os.path.join(self.tweet_tree_path, "reactions")
        self.root_tweet_structure_path = os.path.join(
            self.tweet_tree_path, "structure.json"
        )
        self.tweet_indices_in_tree = {}
        self.node_indices = 0

    def get_tweet_text(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)
            logger.debug(data.get("text"))
        return data.get("text")

    def get_source_tweet(self):
        root_tweet_filepath = (
            os.path.join(self.tweet_tree_path, "source-tweets", self.tweet_tree_idx)
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
        root_tweet_annot_path = os.path.join(self.tweet_tree_path, "annotation.json")
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
    convert_pheme_sequential()
