import os
import logging
from posixpath import join
from tqdm import tqdm
import time

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s:%(lineno)d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

par_dir = os.path.dirname(os.getcwd())


def load9foldData():
    dir_name = os.path.join(par_dir, "all-rnr-annotated-threads")
    sub_dirs = os.listdir(dir_name)

    event_folds = {}
    pbar = tqdm(range(len(sub_dirs)), desc="Event")
    for event_dir in sub_dirs:
        if event_dir.startswith("."):
            continue
        logger.debug("\t-------------------------------------")
        logger.debug("Processing event {}".format(event_dir))
        files = os.listdir(os.path.join(dir_name, event_dir))
        event_folds[event_dir] = []
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
                event_folds[event_dir].append(int(tweet_tree_idx))

    folds_dict = {}
    for i, cur_key in enumerate(event_folds):
        test_identifiers = event_folds[cur_key]
        train_identifiers = []

        for it_key in event_folds:
            if it_key != cur_key:
                train_identifiers.extend(event_folds[it_key])
        logger.info(
            f"Loading fold {i}# {cur_key} \t sample size: train {len(train_identifiers)}, test {len(test_identifiers)}"
        )

        test_fname = os.path.join(
            par_dir, "preprocess/folds9/RNNtestSet_PHEME" + str(i) + "_tree.txt"
        )
        train_fname = os.path.join(
            par_dir, "preprocess/folds9/RNNtrainSet_PHEME" + str(i) + "_tree.txt"
        )
        with open(train_fname, "w") as filehandle:
            filehandle.writelines(
                "%s\n" % tree_idx for tree_idx in list(train_identifiers)
            )

        with open(test_fname, "w") as filehandle:
            filehandle.writelines(
                "%s\n" % tree_idx for tree_idx in list(test_identifiers)
            )
        folds_dict[i] = (train_identifiers, test_identifiers)

    return folds_dict


if __name__ == "__main__":
    load9foldData()
