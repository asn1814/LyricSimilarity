"""
Running this module creates 
"""

import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from collections import Counter

__author__ = "Andrew Nakamoto"

NMOSTCOMMON = 500

def main():
    """ Main entry point of the app """

    # get a pandas dataframe
    SongsCSVDF: pd.DataFrame = getDataFrameSongsCSV()

    # build the corpus
    corpus: [] = buildCorpus(SongsCSVDF)
    
    # get a wordcount of the entire corpus
    wordcounts = Counter(corpus)
    plotCommonWordOccurances(wordcounts, len(SongsCSVDF.index), NMOSTCOMMON)

    mostCommonWords = dict(wordcounts.most_common(NMOSTCOMMON))


def plotCommonWordOccurances(wordcounts: Counter, numSongs: int, n: int):
    plt.figure()
    plt.title(f"Occurances of top {n} words in {numSongs} songs")
    plt.xlabel("Common words")
    plt.ylabel("Occurrances")
    count = 0
    for word, occurances in wordcounts.most_common(n):
        if (count < 5):
            plt.annotate(word, [count, occurances])
        plt.plot(count, occurances, "b-o")
        count += 1
    plt.show()


def getDataFrameSongsCSV() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv('Songs.csv')
    df = df[["Title", "Lyrics"]]
    return df


def buildCorpus(data: pd.DataFrame) -> []: 
    # get stopwords
    # SOURCE: from nltk.corpus import stopwords, nltk.download('stopwords'), stops = stopwords.words('english')
    stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    # build the corpus
    corpus = []
    for index, row in data.iterrows():
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

    return corpus


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()