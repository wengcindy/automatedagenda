import os
import logging
import csv
import pandas as pd

# Gensim
from gensim import similarities
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim import models

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# use files generated fromfrom gensim1.py
if (os.path.exists("/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict")):
   dictionary = corpora.Dictionary.load('/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict')
   # document term matrix
   corpus = corpora.MmCorpus('/Users/cindyweng/Documents/Duke/Automated agenda management/corpus.mm')
   # print("Used files generated from gensim1.py")
else:
   print("Run gensim1.py to generate data set")

def sentenceSimilarity(dictionary, corpus, text):
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

  # load pre-made list of pros/cons
  clean_text = []
  with open("fullText.csv") as csvfile:
      reader = csv.reader(csvfile) 
      for row in reader: # each row is a list
          clean_text.append(row)

  # load pros/cons labels of pre-made list
  proconLabel = []
  with open("proconLabel.csv") as csvfile:
      reader = csv.reader(csvfile) 
      for row in reader: # each row is a list
          proconLabel.append(row)

  # determine if sentence is pro/con based on top 3 sentence similarity matches
  label = ""
  procount = 0
  concount = 0
  for i in range(3):
    # print("\n")
    # print(proconLabel[sims[i][0]])
    # print(clean_text[sims[i][0]])
    # print("\n")
    if proconLabel[sims[i][0]][0] == 'pro':
      procount += 1
    else:
      concount += 1
  # print(text)
  # print(clean_text[sims[i][0]])
  if procount > concount:
    # print('PRO')
    label = "PRO"
  else:
    # print('CON')
    label = "CON"
  # print("\n")
  # print("\n")

  return label

# append pro/con label to existing csv file
transcript = []
labels = []

with open("2019winter.csv") as csvfile:
  reader = csv.reader(csvfile) 
  next(reader)
  for row in reader: 
      joined = ' '.join(row)
      labels.append(sentenceSimilarity(dictionary, corpus, joined))
      
df = pd.read_csv("2019winter.csv")
df['Pros/Cons'] = labels

df.to_csv('2019winterlabeled.csv')


