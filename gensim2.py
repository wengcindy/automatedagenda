import os
import logging
import csv
from gensim import corpora
from gensim import models
from gensim import similarities


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load dictionary, doc term matrix, and LSI model from gensim1.py
dictionary = corpora.Dictionary.load('/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict')
corpus = corpora.MmCorpus('/Users/cindyweng/Documents/Duke/Automated agenda management/corpus.mm')

# initialize LSI model
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=5)

# sentence to check if pro or con
query = " I will lose out on like income tax and other like taxation things when we don't provide undocumented immigrants a path to citizenship special you like those have been here for so long. Money is like in the economy and is not going to support infrastructure. So I think that's another reason to support a pack of citizen."

# convert document into the bag-of-words format, list of (token_id, token_count) tuples
vec_bow = dictionary.doc2bow(query.lower().split())

# convert the query to LSI space
vec_lsi = lsi[vec_bow]  

# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsi[corpus])

# save and load index
# index.save('/Users/cindyweng/Documents/Duke/Automated agenda management/test.index')
# index = similarities.MatrixSimilarity.load('/Users/cindyweng/Documents/Duke/Automated agenda management/test.index')

# perform a similarity query against the corpus
sims = index[vec_lsi]

# (document number, similarity score) 2-tuples
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)

# load pre-made list of pros/cons
clean_text = []
with open("test.csv") as csvfile:
    reader = csv.reader(csvfile) 
    for row in reader: # each row is a list
        clean_text.append(row)

# load pros/cons labels of pre-made list
proconLabel = []
with open("proconLabel.csv") as csvfile:
    reader = csv.reader(csvfile) 
    for row in reader: # each row is a list
        proconLabel.append(row)

for i in range(5):
	print("\n")
	print(proconLabel[sims[i][0]])
	print(clean_text[sims[i][0]])
	print("\n")


