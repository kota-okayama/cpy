"""Generate RecordContainer and save it as yaml file"""

import os
from triwave.file_container import RecordContainer

if __name__ == "__main__":
    dirpath = os.path.dirname(os.path.abspath(__file__))

    rc = RecordContainer()
    # rc.load_tsv(os.path.join(dirpath, "..", "music_20k.csv"), "CID", delimiter=",")
    # rc.load_tsv(os.path.join(dirpath, "..", "persons_5M.csv"), "recid", delimiter=",")
    rc.load_tsv(os.path.join(dirpath, "..", "National_ISBN.tsv"), "cluster_id")

    rc.make_record_random(2000, 60, True)
    rc.save_yaml(os.path.join(dirpath, "output.yaml"))
