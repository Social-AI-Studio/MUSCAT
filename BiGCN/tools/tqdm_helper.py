import tqdm
import time


def myprogress(curr, N, msg, width=10, bars=u"▉▊▋▌▍▎▏ "[::-1], full="█", empty=" "):
    p = curr / N
    nfull = int(p * width)
    return "{}:  {:>3.0%} |{}{}{}| {:>2}/{}".format(
        msg,
        p,
        full * nfull,
        bars[int(len(bars) * ((p * width) % 1))],
        empty * (width - nfull - 1),
        curr,
        N,
    )
