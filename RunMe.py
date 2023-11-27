"""
Module Docstring
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

from collections import Counter

__author__ = "Andrew Nakamoto"


def main():
    """ Main entry point of the app """
    df: pd.DataFrame = pd.read_csv('Songs.csv')
    df = df[["Title", "Lyrics"]]

    corpus = []

    nltk.download('stopwords')
    stops = stopwords.words('english')

    # build the corpus
    for index, row in df.iterrows():
        # get the lyrics of each song
        text = row["Lyrics"]
        if type(text) is not str:
            # some songs do not have lyrics inputted and instead have NaN
            # this skips them
            continue

        # considered a more complex tokenizer but realized it was mostly irrelevant
        text = re.sub("[0-9]+|\||\(|\)", "", text.lower())
        tokens = re.split("\W+", text)

        for word in tokens:
            if word not in stops:
                corpus.append(word)

    print(corpus)
    wordcounts = Counter(corpus)
    print(wordcounts.most_common(40))


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()