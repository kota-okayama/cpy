"""Convert data from version 2.2 to 3.0"""

from triwave.file_container import RecordContainer


# このファイルをrootに配置
if __name__ == "__main__":
    rc = RecordContainer()
    rc.load_file("./benchmark/bib_japan/bibjp_500_a.yaml")
    rc.save_yaml("./bibjp_500_a.yaml")
