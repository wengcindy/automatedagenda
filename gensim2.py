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

    def similarity_query(self, text, topic, section, neutral_cutoff=0.1, best_matches_count=1, verbose=True):
        """
        Perform a similarity query of a text against the entire corpus.
        Returns list of words (as indexes) sorted by similarity.
        :param text: Text
        :return: - Label (pro or con)
                 - Best match across all contexts
                 - Best match among specific topic
                 - Best match in specific topic and section
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

        def get_pro_con_index(index):
            i = index
            while i >= 0 and proConTopicLabel[1][i] == proConTopicLabel[1][index] \
                    and proConTopicLabel[2][i] == proConTopicLabel[2][index] and proConTopicLabel[0][i] == proConTopicLabel[0][index]:
                i -= 1
            return index - i

        def generate_string(index):
            return "%s %s %s %d" % (
                proConTopicLabel[1][sims[index][0]], proConTopicLabel[2][sims[index][0]],
                proConTopicLabel[0][sims[index][0]], get_pro_con_index(sims[index][0]))

        best_match = generate_string(0)
        best_match_similarity = sims[0][1]
        best_match_same_topic = None
        best_match_same_topic_similarity = None
        best_match_same_section = None
        best_match_same_section_similarity = None
        if verbose:
            print("  %f %s %s %s %s" % (sims[0][1], proConTopicLabel[0][sims[0][0]], proConTopicLabel[1][sims[0][0]], proConTopicLabel[2][sims[0][0]], proConTopicLabel[3][sims[0][0]]))  # DEBUG
        matches_counted = 0
        for i in range(len(sims)):
            if proConTopicLabel[1][sims[i][0]] == topic:
                if proConTopicLabel[2][sims[i][0]] == section:
                    if best_match_same_section is None:
                        best_match_same_section = generate_string(i)
                        best_match_same_section_similarity = sims[i][1]
                        if best_match_same_topic is None:
                            best_match_same_topic = generate_string(i)
                            best_match_same_topic_similarity = sims[i][1]
                    if sims[i][1] >= neutral_cutoff and matches_counted < best_matches_count:
                        matches_counted += 1
                        if proConTopicLabel[0][sims[i][0]] == 'pro':
                            procount += sims[i][1]
                        else:
                            concount += sims[i][1]
                    if verbose:
                        print("  %f %s %s %s %s" % (sims[i][1], proConTopicLabel[0][sims[i][0]], proConTopicLabel[1][sims[i][0]], proConTopicLabel[2][sims[i][0]], proConTopicLabel[3][sims[i][0]]))  # DEBUG
                elif best_match_same_topic is None:
                    best_match_same_topic = generate_string(i)
                    best_match_same_topic_similarity = sims[i][1]
                    if verbose:
                        print("  %f %s %s %s %s" % (sims[i][1], proConTopicLabel[0][sims[i][0]], proConTopicLabel[1][sims[i][0]], proConTopicLabel[2][sims[i][0]], proConTopicLabel[3][sims[i][0]]))  # DEBUG

        if procount > concount and procount + concount > 0:
            label = "pro"
        else:
            label = "con" if procount + concount > 0 else "neutral"

        return label, best_match, best_match_similarity, best_match_same_topic, best_match_same_topic_similarity, best_match_same_section, best_match_same_section_similarity

