"""
Running this module creates 
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import gensim
import itertools
import networkx as nx
#from sklearn.decomposition import KernelPCA

from collections import Counter

__author__ = "Andrew Nakamoto"

NMOSTCOMMON: int = 100
PROCESSEDCSVFILEPATH: str = "RelativeFrequencies.csv"
LDAFILEPATH: str = "LDA.csv"
LYRICS: str = "Lyrics"
SONGMETADATA: list = ["Artist", "Title", LYRICS]
NUMTOPICS = 40
NUMLDAPASSES = 100

def main():
    """ Main entry point of the app """

    # get a pandas dataframe
    SongsCSVDF: pd.DataFrame = getDataFrameSongsCSV()
    
    #useBagOfWordsPCA(SongsCSVDF)

    useLDA(SongsCSVDF)

def useBagOfWordsPCA(SongsCSVDF: pd.DataFrame):
    # build the corpus
    corpus: [] = buildCorpus(SongsCSVDF)
    
    # get a wordcount of the entire corpus
    wordcounts = Counter(corpus)
    # plotCommonWordOccurances(wordcounts, len(SongsCSVDF.index), NMOSTCOMMON)

    # can take a while due to buildRelativeOccurances, so do it once and then use the saved csv
    mostCommonWords: dict = dict(wordcounts.most_common(NMOSTCOMMON))
    relativeDF = buildRelativeOccurances(mostCommonWords, SongsCSVDF)
    relativeDF.to_csv(PROCESSEDCSVFILEPATH, index=False, encoding="utf-8")

    relativeDF: pd.DataFrame = pd.read_csv(PROCESSEDCSVFILEPATH, encoding="utf-8")

    plotPCA(relativeDF)


def useLDA(SongsCSVDF: pd.DataFrame): 
    # https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
    # basic processing on the lyrics
    lyrics_set = list(SongsCSVDF[LYRICS])
    processed_lyrics = list()
    for elt in lyrics_set:
        processed_lyrics.append(removeStopsAndStem(tokenize(elt)))

    # The Dictionary() function traverses texts, assigning a unique integer id to each unique token while also collecting word counts and relevant statistics. To see each token’s unique integer id, try print(dictionary.token2id).
    dictionary = gensim.corpora.Dictionary(processed_lyrics)

    # The doc2bow() function converts dictionary into a bag-of-words. The resulting corpus is a list of vectors equal to the number of documents.
    corpus = [dictionary.doc2bow(text) for text in processed_lyrics]

    # The LdaModel class is described in detail in the gensim documentation.
    print("Beginning LDA training")
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUMTOPICS, id2word = dictionary, passes=NUMLDAPASSES)
    print("LDA model trained")

    # Our LDA model is now stored as ldamodel. We can review our topics with the print_topic and print_topics methods
    # https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
    """df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=SongsCSVDF)
    
    # Format
    df_dominant_topic = df_topic_sents_keywords#.reset_index()
    df_dominant_topic.columns = SONGMETADATA + ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    print(df_dominant_topic)
    df_dominant_topic.to_csv(LDAFILEPATH, index=False, encoding="utf-8")"""

    # my work
    df_with_LDA_vecs = LDA_to_vecs(ldamodel, corpus, SongsCSVDF)
    df_with_LDA_vecs.columns = SONGMETADATA + ["LDA Vector"]
    plotPCA(df_with_LDA_vecs, f"PCA on LDA vectors with {NUMTOPICS} topics and {NUMLDAPASSES} passes")


def LDA_to_vecs(ldamodel: gensim.models.LdaModel, corpus, texts: pd.DataFrame) -> pd.DataFrame:
    df_with_vec = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list  
        # print(row)
        vec = np.zeros(NUMTOPICS)
        for (topic_num, prop_topic) in row:
            vec[topic_num] = prop_topic
        
        df_with_vec = pd.concat([df_with_vec, pd.Series([vec])], axis=0, ignore_index=True)

    df_with_vec.columns = ["LDA Vector"]

    return pd.concat([texts, df_with_vec], ignore_index=True, axis=1)


def format_topics_sentences(ldamodel: gensim.models.LdaModel, corpus, texts):
    # Init output
    sent_topics_df: pd.DataFrame = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                #sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                series_row = pd.Series([int(topic_num), round(prop_topic,4), topic_keywords])
                sent_topics_df = pd.concat([sent_topics_df, series_row], ignore_index=True, axis=1)
            else:
                break
    
    sent_topics_df = sent_topics_df.T
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    # contents = pd.Series(texts)
    sent_topics_df = pd.concat([texts, sent_topics_df], axis=1, ignore_index=True)
    return(sent_topics_df)


def removeStopsAndStem(tokens: list) -> list:
    stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stops.append("urlcopyembedcopy")

    stemmer = nltk.stem.PorterStemmer()
    
    result = []
    for token in tokens:
        if token not in stops and token.find("embedshare") == -1:
            #result.append(stemmer.stem(token)) # use the stemmer
            result.append(token) # dont use the stemmer
    return result


def plotPCA(df: pd.DataFrame, title: str = "PCA Visualization"):
    arr: np.ndarray = df.drop(columns=SONGMETADATA).to_numpy()
    if type(arr[0]) is np.ndarray: # if sent from LDA, need this to clean it up
        newarr = np.zeros([arr.shape[0], NUMTOPICS])
        for idx, elt in enumerate(arr):
            newarr[idx] = elt[0]
        arr = newarr
    # normalize and standardize the vectors
    standardizedData = (arr - arr.mean(axis=0)) / arr.std(axis=0)
    covarianceMatrix = np.cov(standardizedData, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1] 
    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
    plotEigenvalues(sorted_eigenvalues)

    k = 2 # select the number of principal components
    reduced_data = np.matmul(standardizedData, sorted_eigenvectors[:,:k]) # transform the original data
    plt.figure()
    plt.ylim(top=5, bottom=-5)
    for index in range(len(reduced_data)):
        x, y = reduced_data[index]
        if (df.iloc[index]["Artist"] == "Taylor Swift"):
            plt.plot(x, y, "b-o", label="Taylor Swift")
            plt.annotate(df.iloc[index]["Title"], [x, y])
        elif (df.iloc[index]["Artist"] == "Frank Sinatra"):
            plt.plot(x, y, "r-o")
        elif (df.iloc[index]["Artist"] == "Ed Sheeran"):
            plt.plot(x, y, "m-o")
        else:
            plt.plot(x, y, "c-o")

    plt.title(title)
    plt.show()
    

def plotEigenvalues(eigenvalues: np.ndarray):
    plt.figure()
    plt.title(f"Eigenvalues of the covariance matrix")
    plt.xlabel("Eigenvalue e")
    plt.ylabel("Magnitude of e")
    eigenvalues = eigenvalues.copy()
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]
    print(eigenvalues)
    plt.plot(np.arange(len(eigenvalues)), eigenvalues)
    plt.show()


def buildRelativeOccurances(mostCommonWords: dict, data: pd.DataFrame) -> pd.DataFrame:
    # set up the dataframe
    cols = SONGMETADATA.copy() # must copy
    wordsToTrack = list(mostCommonWords.keys())
    cols.extend(wordsToTrack)
    df: pd.DataFrame = pd.DataFrame(columns=cols)

    # add each song to the DataFrame
    for index, row in data.iterrows():
        print(f"Processing song {index}")
        # get tokens of the song
        tokens = tokenize(row[LYRICS])
        wordcounts: Counter = Counter(tokens)
        # build up the new row to add using the song
        newRow: pd.Series = pd.Series()
        for feature in SONGMETADATA:
            newRow[feature] = row[feature]
        for word in wordsToTrack:
            i: float = wordcounts.get(word)
            if i is None:
                i = 0
            newRow[word] = i / mostCommonWords.get(word)
        # add the new row to the DataFrame
        df = pd.concat([df, newRow.to_frame().T], ignore_index=True)

    return df


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
    df = df[["Artist", "Title", "Lyrics"]] # equivalent to SONGMETADATA
    return df


def buildCorpus(data: pd.DataFrame) -> []: 
    # get stopwords
    # SOURCE: from nltk.corpus import stopwords, nltk.download('stopwords'), stops = stopwords.words('english')
    stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    # build the corpus
    corpus = []
    for index, row in data.iterrows():
        # get the lyrics of each song
        text = row[LYRICS]
        if type(text) is not str:
            # some songs do not have lyrics inputted and instead have NaN
            # this skips them
            continue

        tokens = tokenize(text)

        for word in tokens:
            if word not in stops:
                corpus.append(word)

    return corpus


def tokenize(text: str) -> []:
    # considered a more complex tokenizer but realized it was mostly irrelevant
    text = re.sub("[0-9]+|\||\(|\)", "", text.lower())
    return re.split("\W+", text)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()