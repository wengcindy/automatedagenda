import json
import numpy
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Word2Vec
from gensim.models import Phrases

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



# read pros/cons file
with open("agendas.json", 'r') as myfile:
    data=myfile.read()

# parse pros/cons file
obj = json.loads(data)
pros = []
cons = []
proconLabel = []

for i in obj:
	for j in range(len(obj[i])):
		pro = ""
		con = ""
		for k in range(len(obj[i][j]['pro'])):
			pro += obj[i][j]['pro'][k] + " "

		for k in range(len(obj[i][j]['con'])):
			con += obj[i][j]['con'][k] + " "
		pros.append(pro)
		cons.append(con)
		# print(pro)
		# print("\n")
		# print(con)
		# print("\n")
		proconLabel.append('pro')
		proconLabel.append('con')

allWords = pros + cons
proTokenized = [word_tokenize(i) for i in pros]
conTokenized = [word_tokenize(i) for i in cons]
proKeywords = []
conKeywords = []

stop_words = stopwords.words('english')

# get pro sentence keywords
for sentences in proTokenized:
	proKeywords.append([word for word in sentences if word not in stop_words])

# get con sentence keywords
for sentences in conTokenized:
	conKeywords.append([word for word in sentences if word not in stop_words])

# combine pro and con keywords
allKeywords = proKeywords + conKeywords

# remove punctuation
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
allKeywords = list(sent_to_words(allKeywords))

# save pros/cons as csv
arr = numpy.asarray(allWords)
numpy.savetxt("fullText.csv", arr, delimiter=",",fmt='%s')
arr = numpy.asarray(allKeywords)
numpy.savetxt("allKeywords.csv", arr, delimiter=",",fmt='%s')
arr = numpy.asarray(proconLabel)
numpy.savetxt("proconLabel.csv", arr, delimiter=",",fmt='%s')

# from pprint import pprint  # pretty-printer
# pprint(allKeywords)

# save dictionary (each unique term given index)
dictionary = corpora.Dictionary(allKeywords)
dictionary.save('/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict')

# convert to document term matrix and save
corpus = [dictionary.doc2bow(text) for text in allKeywords]
corpora.MmCorpus.serialize('/Users/cindyweng/Documents/Duke/Automated agenda management/corpus.mm', corpus)


