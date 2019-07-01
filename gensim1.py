"""
Parses pre-made agenda and labels agenda items with topic and pro/con and saves as csv.
Tokenize, process, and extract keywords from agenda items, and create dictionary and document term matrix from keywords.
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

import warnings
warnings.filterwarnings("ignore")



# read pre-made agenda file
with open("agendas.json", 'r') as myfile:
    data=myfile.read()

# parse agenda file
obj = json.loads(data)
#pros = []
#cons = []
sentences = []  # Stores each pro/con as a separate element in the list
# label if item is pro or con
proconLabel = []
# label topic of item
topicLabel = []
# label section of item
sectionLabel = []

#allWords = []

for i in obj:
	for j in range(len(obj[i])):
		#for k in range(len(obj[i][j]['pro'])):
		for pro in obj[i][j]['pro']:
		#for pro in [' '.join(obj[i][j]['pro'])]:
			sentences.append(pro.lower())
			topicLabel.append(i)
			sectionLabel.append("A" + str(j+1))
			proconLabel.append("pro")
		for con in obj[i][j]['con']:
		#for con in [' '.join(obj[i][j]['con'])]:
			sentences.append(con.lower())
			topicLabel.append(i)
			sectionLabel.append("A" + str(j+1))
			proconLabel.append("con")

		"""pro = ""
		con = ""
		for k in range(len(obj[i][j]['pro'])):
			pro += obj[i][j]['pro'][k].lower() + " "
		for k in range(len(obj[i][j]['con'])):
			con += obj[i][j]['con'][k].lower() + " "
		if i == "electoralReform":
			topicLabel.append("electoralReform")
			topicLabel.append("electoralReform")
			sectionLabel.append("A"+str(j+1))
			sectionLabel.append("A"+str(j+1))
		elif i == "campaignFinanceReform":
			topicLabel.append("campaignFinanceReform")
			topicLabel.append("campaignFinanceReform")
			sectionLabel.append("A"+str(j+1))
			sectionLabel.append("A"+str(j+1))
		elif i == "immigration":
			topicLabel.append("immigration")
			topicLabel.append("immigration")
			sectionLabel.append("A"+str(j+1))
			sectionLabel.append("A"+str(j+1))
		elif i == "microworkers":
			topicLabel.append("microworkers")
			topicLabel.append("microworkers")
			sectionLabel.append("A"+str(j+1))
			sectionLabel.append("A"+str(j+1))
		elif i == "alicesClass":
			topicLabel.append("alicesClass")
			topicLabel.append("alicesClass")
			sectionLabel.append("A"+str(j+1))
			sectionLabel.append("A"+str(j+1))
		pros.append(pro)
		cons.append(con)
		allWords.append(pro)
		allWords.append(con)
		# print(pro)
		# print("\n")
		# print(con)
		# print("\n")
		proconLabel.append('pro')
		proconLabel.append('con')"""


#proTokenized = [word_tokenize(i) for i in pros]
#conTokenized = [word_tokenize(i) for i in cons]
sentences_tokenized = [word_tokenize(i) for i in sentences]
#proKeywords = []
#conKeywords = []
allKeywords = []

stop_words = stopwords.words('english')
stop_words.extend(['could', 'would', 'and'])

# get pro sentence keywords
#for sentences in proTokenized:
#	proKeywords.append([word for word in sentences if word not in stop_words])

# get con sentence keywords
#for sentences in conTokenized:
#	conKeywords.append([word for word in sentences if word not in stop_words])
for sentence in sentences_tokenized:
	allKeywords.append([word for word in sentence if word not in stop_words])

# combine pro and con keywords
#allKeywords = proKeywords + conKeywords

# remove punctuation
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
allKeywords = list(sent_to_words(allKeywords))
#allWords = list(sent_to_words(allWords))

#allWordsJoined = []
#print(allWords)
#for sentence in allWords:
	#print(len(sentence))
#	allWordsJoined.append(' '.join(sentence))

# print(allWordsJoined)

# save pros/cons as csv
#arr = numpy.asarray(allWordsJoined)
#numpy.savetxt("fullText.csv", arr, delimiter=",",fmt='%s')
arr = numpy.asarray(allKeywords)
numpy.savetxt("allKeywords.csv", arr, delimiter=",",fmt='%s')
#arr = numpy.array([proconLabel, topicLabel, sectionLabel, allWordsJoined])
allWords = [' '.join(words) for words in list(sent_to_words(sentences))]
arr = numpy.array([proconLabel, topicLabel, sectionLabel, allWords])
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



