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


st = SimilarityTester()
print(st.similarity_query("Survvng on this one was evn more that I guess in the last one if you're a witness to a crime, I think we really want to encourage those people to come forward just because I think it's more necessary and in solvng crimes in my kind of getting to the bottom of whether or not so just report."))


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# use files generated fromfrom gensim1.py
if (os.path.exists("/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict")):
   dictionary = corpora.Dictionary.load('/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict')
   # document term matrix
   corpus = corpora.MmCorpus('/Users/cindyweng/Documents/Duke/Automated agenda management/corpus.mm')
   # print("Used files generated from gensim1.py")
else:
   print("Run gensim1.py to generate data set")

# topic can be one of the agenda labels: electoralReform, campaignFinanceReform, immigration
def sentenceSimilarity(dictionary, corpus, text, topic, section):
  # initialize model - go through text once and compute document frequencies of all its features
  tfidf = models.TfidfModel(corpus)

  # transform corpus from BOW to TfIdf real-valued weights representation
  corpus_tfidf = tfidf[corpus]
  # for doc in corpus_tfidf:
  #     print(doc)

  # initialize an LSI transformation
  lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=44)
  corpus_lsi = lsi[corpus_tfidf] 
  # for doc in corpus_lsi:
  # 	print(doc)

  # save model
  # lsi.save('/tmp/model.lsi')
  # lsi = models.LsiModel.load('/tmp/model.lsi')

  # convert the query (sentence from transcript) to LSI space
  doc = text
  vec_bow = dictionary.doc2bow(doc.split())
  vec_lsi = lsi[vec_bow]

  # transform corpus to LSI space and index it
  index = similarities.MatrixSimilarity(lsi[corpus])  

  # save and load index
  # index.save('/Users/cindyweng/Documents/Duke/Automated agenda management/test.index')
  # index = similarities.MatrixSimilarity.load('/Users/cindyweng/Documents/Duke/Automated agenda management/test.index')

  # perform a similarity query against the corpus
  sims = index[vec_lsi]
  sims = sorted(enumerate(sims), key=lambda item: -item[1])
  # print(sims)

  # load pre-made agenda of pros/cons
  clean_text = []
  with open("fullText.csv") as csvfile:
      reader = csv.reader(csvfile) 
      for row in reader: # each row is a list
          clean_text.append(row)

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
  if section.startswith('A'):
    if procount > concount:
      label = "pro"
    else:
      label = "con"

  return label


# save pro/con label to csv file with other stats
"""transcript = []
labels = []

filename = "2019winter"
topic = "immigration"
with open(filename + ".csv") as csvfile:
  reader = csv.reader(csvfile) 
  next(reader)
  for row in reader: 
      joined = ' '.join(row)
      labels.append(sentenceSimilarity(dictionary, corpus, row[1], topic, row[6]))
      
df = pd.read_csv(filename + ".csv")
df['Pro or con'] = labels

df.to_csv(filename + "labeled.csv")
"""

