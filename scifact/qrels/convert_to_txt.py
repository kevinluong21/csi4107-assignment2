import pandas as pd

# This script is solely intended to convert tsv files from the dataset into txt files that are formatted for trec_eval

df = pd.read_csv(r"scifact/qrels/test.tsv", sep="\t")
df.insert(value="0", loc=1, column="constant")
df.to_csv(r"scifact/qrels/test.txt", header=False, index=False, sep=" ")