"""
Creates LSI model from dictionary and DTM, using it for sentence similarity comparisons to the transcript.
Uses latent semantic indexing, an info-retrieving technique that uses latent semantic analysis.
Run gensim1.py before running this.
"""

import os
import logging
import csv
import pandas as pd

# Gensim
from gensim import similarities
from gensim import corpora
from gensim import models


class SimilarityTester:
    def __init__(self):
        #if (os.path.exists("/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict")):
        if (os.path.exists("dictionary.dict")):
            #dictionary = corpora.Dictionary.load('/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict')
            self.dictionary = corpora.Dictionary.load('dictionary.dict')
            # document term matrix
            #corpus = corpora.MmCorpus('/Users/cindyweng/Documents/Duke/Automated agenda management/corpus.mm')
            self.corpus = corpora.MmCorpus('corpus.mm')
            # print("Used files generated from gensim1.py")
        else:
            raise RuntimeError("Run gensim1.py to generate data set")

        # initialize model - go through text once and compute document frequencies of all its features
        self.tfidf = models.TfidfModel(self.corpus)

        # transform corpus from BOW to TfIdf real-valued weights representation
        self.corpus_tfidf = self.tfidf[self.corpus]
        # for doc in corpus_tfidf:
        #     print(doc)

        # initialize an LSI transformation
        self.lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=44)
        # self.corpus_lsi = lsi[self.corpus_tfidf]
        # for doc in corpus_lsi:
        # 	print(doc)

    def similarity_query(self, text, topic, section):
        """
        Perform a similarity query of a text against the entire corpus.
        Returns list of words (as indexes) sorted by similarity.
        :param text: Text
        :return: List of words
        """
        # convert the query (sentence from transcript) to LSI space
        doc = text
        vec_bow = self.dictionary.doc2bow(doc.split())
        vec_lsi = self.lsi[vec_bow]

        # transform corpus to LSI space and index it
        index = similarities.MatrixSimilarity(self.lsi[self.corpus])

        # save and load index
        # index.save('/Users/cindyweng/Documents/Duke/Automated agenda management/test.index')
        # index = similarities.MatrixSimilarity.load('/Users/cindyweng/Documents/Duke/Automated agenda management/test.index')

        # perform a similarity query against the corpus
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        # print(sims)

        # load pros/cons and topic labels of pre-made list
        proConTopicLabel = []
        with open("proConTopicLabel.csv") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader: # each row is a list
                proConTopicLabel.append(row)

        # determine if sentence is pro/con based on top 3 sentence similarity matches
        # only compares sentence to agenda item of the same topic and same section
        label = ""
        procount = 0
        concount = 0

        for i in range(len(sims)):
            if proConTopicLabel[1][sims[i][0]] == topic and proConTopicLabel[2][sims[i][0]] == section:
                # print("\n")
                # print(proconLabel[sims[i][0]])
                # print(clean_text[sims[i][0]])
                # print("\n")
                # print(proConTopicLabel[1][sims[i][0]])
                # print(proConTopicLabel[0][sims[i][0]])
                if proConTopicLabel[0][sims[i][0]] == 'pro':
                    procount += 1
                else:
                    concount += 1
                break
        if procount > concount:
            label = "pro"
        else:
            label = "con"

        return label

