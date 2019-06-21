"""
Parses pre-made agenda and labels agenda items with topic and pro/con and saves as csv.
Tokenize, process, and extract keywords from agenda items, and create dictionary 
and document term matrix from keywords.
Run gensim1.py before running gensim2.py.
"""

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

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



# read pre-made agenda file
with open("agendas.json", 'r') as myfile:
    data=myfile.read()

# parse agenda file
obj = json.loads(data)
pros = []
cons = []
# label if item is pro or con
proconLabel = []
# label topic of item
topicLabel = []

for i in obj:
	for j in range(len(obj[i])):
		pro = ""
		con = ""
		for k in range(len(obj[i][j]['pro'])):
			pro += obj[i][j]['pro'][k].lower() + " "

		for k in range(len(obj[i][j]['con'])):
			con += obj[i][j]['con'][k].lower() + " "
		if i == "electoralReform":
			topicLabel.append("electoralReform")
			topicLabel.append("electoralReform")
		elif i == "campaignFinanceReform":
			topicLabel.append("campaignFinanceReform")
			topicLabel.append("campaignFinanceReform")
		elif i == "immigration":
			topicLabel.append("immigration")
			topicLabel.append("immigration")
		elif i == "microworkers":
			topicLabel.append("microworkers")
			topicLabel.append("microworkers")
		elif i == "alicesClass":
			topicLabel.append("alicesClass")
			topicLabel.append("alicesClass")
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
stop_words.extend(['could', 'would', 'and'])

# get pro sentence keywords
for sentences in proTokenized:
	proKeywords.append([word for word in sentences if word not in stop_words])

# get con sentence keywords
for sentences in conTokenized:
	conKeywords.append([word for word in sentences if word not in stop_words])

# combine pro and con keywords
allKeywords = proKeywords + conKeywords
print(allKeywords)

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

arr = numpy.array([proconLabel, topicLabel])
numpy.savetxt("proConTopicLabel.csv", arr, delimiter=",",fmt='%s')

# from pprint import pprint  # pretty-printer
# pprint(allKeywords)

# save dictionary (each unique term given index)
dictionary = corpora.Dictionary(allKeywords)
#dictionary.save('/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict')
dictionary.save('dictionary.dict')

# convert to document term matrix (unique index, term frequency) and save
corpus = [dictionary.doc2bow(text) for text in allKeywords]
#corpora.MmCorpus.serialize('/Users/cindyweng/Documents/Duke/Automated agenda management/corpus.mm', corpus)
corpora.MmCorpus.serialize('corpus.mm', corpus)