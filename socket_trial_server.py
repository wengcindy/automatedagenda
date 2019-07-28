import eventlet
import socketio
import os
import logging
import csv
import numpy as np
import pandas as pd
from pprint import pprint
import sys
import json

# Disable gensim warnings
import warnings
warnings.filterwarnings("ignore")
# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim import similarities
from gensim import corpora
from gensim import models

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


sio = socketio.Server()
#app = socketio.WSGIApp(sio, static_files={
#    '/': {'content_type': 'text/html', 'filename': 'index.html'}
#})
app = socketio.WSGIApp(sio)


# --------------------------------


NUM_RESULTS_SAME_TOPIC = 3  # number of results to be displayed from the same topic
NEUTRAL_CUTOFF = 0.1  # Minimum similarity for an argument to be counted towards the label
PRO = 'Pro'
CON = 'Con'
NEUTRAL = 'Neutral'


proConTopicLabel = []
data = {}
sentences = []  # All pros and cons listed by IDs (for dictionary)
topics = []  # Topics of each sentences corresponding to sentences
sections = []  # Section index (as integers)
point_subindex = []  # Index of this argument within pros/cons of current section (0-indexed)
labels = []  # "pro" or "con"
arguments = {}  # {topic: {section_str: {pros: [ids], cons: [ids]}}
dictionary = None

st = None  # SimilarityTester


def initialize(args):
    """
    Initialize the dictionary and corpus.
    (This is what the old gensim1.py does)
    :param args: All pros and cons points in the format of: {topic: [[[pros0], [cons0]], [[pros1], [cons1]], ...]}
    """
    global data, sentences, topics, sections, point_subindex
    data = args

    for topic, topics_points in data.items():
        for section in range(len(topics_points)):
            pros, cons = topics_points[section]
            for i in range(len(pros)):
                sentences.append(pros[i])
                topics.append(topic)
                sections.append(section)
                point_subindex.append(i)
                labels.append(PRO)
            for i in range(len(cons)):
                sentences.append(cons[i])
                topics.append(topic)
                sections.append(section)
                point_subindex.append(i)
                labels.append(CON)


    sentences_tokenized = [word_tokenize(i) for i in sentences]
    allKeywords = []

    stop_words = stopwords.words('english')
    stop_words.extend(['could', 'would', 'and'])

    for sentence in sentences_tokenized:
        allKeywords.append([word for word in sentence if word not in stop_words])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    allKeywords = list(sent_to_words(allKeywords))

    global dictionary
    dictionary = corpora.Dictionary(allKeywords)
    corpus = [dictionary.doc2bow(text) for text in allKeywords]
    corpora.MmCorpus.serialize('corpus.mm', corpus)

    global st
    st = SimilarityTester()


#def construct_arguments():
    """
    Construct the arguments dict.
    """
    """global proConTopicLabel
    proConTopicLabel = []
    with open("proConTopicLabel.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader: # each row is a list
            proConTopicLabel.append(row)
    for i in range(len(proConTopicLabel[0])):
        topic = proConTopicLabel[1][i]
        section = proConTopicLabel[2][i]
        text = proConTopicLabel[3][i]
        label = proConTopicLabel[0][i]
        if topic not in arguments:
            arguments[topic] = {}
        if section not in arguments[topic]:
            arguments[topic][section] = {}
        if label not in arguments[topic][section]:
            arguments[topic][section][label] = []
        arguments[topic][section][label].append(i)"""


def get_topic(index):
    """
    Given an index of a sentence in the corpus, find its corresponding topic (e.g. "immigration").
    """
    return proConTopicLabel[1][index]


def get_pro_con_index(index):
    """
    Given an index of a sentence in the corpus, find its corresponding index among EITHER the pros OR the cons.
    (e.g. con 1, con 2, etc)
    """
    entries = [arguments[topic][section][label]
               for topic in arguments 
               for section in arguments[topic] 
               for label in arguments[topic][section]
               if index in arguments[topic][section][label]]
    return entries[0].index(index) + 1


def get_point_number(index):
    """
    Given an index of a sentence in the corpus, find the corresponding point number as listed in the agenda.
    (e.g. Pros are 1.1, 1.2, Cons are 1.3, 1.4)
    :return: "section number.point number"
    """
    if index is None:
        return None
    """entries = [(topic, section, label, arguments[topic][section][label])
               for topic in arguments 
               for section in arguments[topic] 
               for label in arguments[topic][section]
               if index in arguments[topic][section][label]]
    topic, section, label, indexes = entries[0]
    index_in_label = indexes.index(index) + 1
    return "%d.%d" % (
        int(section[1:]), 
        (index_in_label if label == 'pro'
         else index_in_label + len(arguments[topic][section]['pro'])))"""
    #return "%d.%d" % (sections[index], (point_subindex[index] + 1 if labels[index] == 'pro'
    #                                    else len(data[topics[index]][sections[index]][0]) + point_subindex[index] + 1))
    return "%s %d.%d" % (labels[index], sections[index], point_subindex[index] + 1)


class SimilarityTester:
    def __init__(self):
        #if (os.path.exists("/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict")):
        """if (os.path.exists("dictionary.dict")):
            #dictionary = corpora.Dictionary.load('/Users/cindyweng/Documents/Duke/Automated agenda management/dictionary.dict')
            self.dictionary = corpora.Dictionary.load('dictionary.dict')
            # document term matrix
            #corpus = corpora.MmCorpus('/Users/cindyweng/Documents/Duke/Automated agenda management/corpus.mm')
            self.corpus = corpora.MmCorpus('corpus.mm')
            # print("Used files generated from gensim1.py")
        else:
            raise RuntimeError("Run gensim1.py to generate data set")"""
        self.dictionary = dictionary
        self.corpus = corpora.MmCorpus('corpus.mm')

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
        #     print(doc)

        # load pros/cons and topic labels of pre-made list
        #construct_arguments()

    def similarity_query(self, text, topic, section, num_results=1, neutral_cutoff=NEUTRAL_CUTOFF, best_matches_count=1, verbose=True):
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

        # determine if sentence is pro/con based on top 3 sentence similarity matches
        # only compares sentence to agenda item of the same topic and same section
        label = ""
        procount = 0
        concount = 0

        def generate_string(index):
            """
            Convert a sentence index to string in old format, e.g. "immigration A3 pro 2".
            """

            if index is None:
               return None
            return ("%s %s %s %d" % (
                #proConTopicLabel[1][index], proConTopicLabel[2][index],
                #proConTopicLabel[0][index], get_pro_con_index(index))
                topics[index], sections[index], labels[index], point_subindex[index] + 1)).replace("Pro", "pro").replace("Con", "con")

        sims_same_topic = [(index, sim) for index, sim in sims if topics[index] == topic]
        sims_same_section = [(index, sim) for index, sim in sims_same_topic if sections[index] == section]

        best_match = sims[0][0]  #generate_string(0)
        best_match_similarity = sims[0][1]
        #best_match_same_topic = sims_same_topic[0][0] if sims_same_topic else None
        #best_match_same_topic_similarity = sims_same_topic[0][1] if sims_same_topic else None
        best_matches_same_topic = [index for index, sim in sims_same_topic][:num_results]
        best_matches_same_topic_similarity = [sim for index, sim in sims_same_topic][:num_results]
        best_match_same_section = sims_same_section[0][0] if sims_same_section else None
        best_match_same_section_similarity = sims_same_section[0][1] if sims_same_section else None
        if verbose:
            tmp = [('Overall', best_match, best_match_similarity), 
                   #('Same topic', best_match_same_topic, best_match_same_topic_similarity),
                   ('Same section', best_match_same_section, best_match_same_section_similarity)]
            for string, index, sim in tmp:
                print("  %s: %f %s %s %s %s" % (
                    string,
                    sim, 
                    topics[index], 
                    sections[index], 
                    labels[index], 
                    point_subindex[index] + 1))
            print("  Same topic:")
            for i in range(len(best_matches_same_topic)):
                print("    %f %s %s %s %s" % (
                    best_matches_same_topic_similarity[i], 
                    topics[best_matches_same_topic[i]], 
                    sections[best_matches_same_topic[i]], 
                    labels[best_matches_same_topic[i]], 
                    point_subindex[best_matches_same_topic[i]] + 1))
            #print("  %f %s %s %s %s" % (sims[0][1], proConTopicLabel[0][sims[0][0]], proConTopicLabel[1][sims[0][0]], proConTopicLabel[2][sims[0][0]], proConTopicLabel[3][sims[0][0]]))  # DEBUG
        
        sims_same_section_counted = sims_same_section[:best_matches_count]
        pros_counted = [(index, sim) for index, sim in sims_same_section_counted 
                        if sim > neutral_cutoff and labels[index] == 'Pro']
        cons_counted = [(index, sim) for index, sim in sims_same_section_counted 
                        if sim > neutral_cutoff and labels[index] == 'Con']
        procount = sum([sim for index, sim in pros_counted])
        concount = sum([sim for index, sim in cons_counted])

        #matches_counted = 0
        """for i in range(len(sims)):
            if proConTopicLabel[1][sims[i][0]] == topic:
                if proConTopicLabel[2][sims[i][0]] == section:
                    if best_match_same_section is None:
                        best_match_same_section = sims[i][0] #generate_string(i)
                        best_match_same_section_similarity = sims[i][1]
                        if best_match_same_topic is None:
                            best_match_same_topic = sims[i][0] #generate_string(i)
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
                    best_match_same_topic = sims[i][0] #generate_string(i)
                    best_match_same_topic_similarity = sims[i][1]
                    if verbose:
                        print("  %f %s %s %s %s" % (sims[i][1], proConTopicLabel[0][sims[i][0]], proConTopicLabel[1][sims[i][0]], proConTopicLabel[2][sims[i][0]], proConTopicLabel[3][sims[i][0]]))  # DEBUG"""


        if procount > concount and procount + concount > 0:
            label = PRO
        else:
            label = CON if procount + concount > 0 else NEUTRAL

        return {
            "label": label,
            "best_match": get_point_number(best_match), #ret[1],
            "best_match_topic": topics[best_match],
            "best_match_old": generate_string(best_match),
            "best_match_similarity": best_match_similarity.item(),
            # TODO: Convert the following to best_matches_same_topic
            #"best_match_same_topic": get_point_number(best_match_same_topic), #ret[3],
            #"best_match_same_topic_topic": topics[best_match_same_topic],
            #"best_match_same_topic_old": generate_string(best_match_same_topic),
            #"best_match_same_topic_similarity": None if best_match_same_topic_similarity is None else best_match_same_topic_similarity.item(),
            "best_matches_same_topic": [get_point_number(match) for match in best_matches_same_topic], #ret[3],
            "best_matches_same_topic_topic": [topics[match] for match in best_matches_same_topic],
            "best_matches_same_topic_old": [generate_string(match) for match in best_matches_same_topic],
            "best_matches_same_topic_similarity": [None if match is None else match.item() for match in best_matches_same_topic_similarity],
            
            "best_match_same_section": get_point_number(best_match_same_section), #ret[5],
            "best_match_same_section_topic": topics[best_match_same_section],
            "best_match_same_section_old": generate_string(best_match_same_section),
            "best_match_same_section_similarity": None if best_match_same_section_similarity is None else best_match_same_section_similarity.item()
        }


# --------------------------------


database = {}  # Map rooms to set of attributes (TBD)


@sio.event
def init(sid, data):
    initialize(data['data'])


@sio.event
def newRoom(sid, data):
    room_name = data['roomName']
    if room_name not in database:
        room = database[room_name] = {}
        room['topic'] = data['agendaName']
        room['sectionIndex'] = 0
        print('new room')
        print(room)


@sio.event
def advanceAgenda(sid, data):
    if data['roomName'] not in database:
        sio.emit('similarityError', {"msg": "Room not initialized"})
        return
    room = database[data['roomName']]
    room['sectionIndex'] = data['agendaIndex']
    room['section'] = "A" + str(room['sectionIndex'])
    print('advance agenda')
    print(room)


@sio.event
def endChat(sid, data):
    if data['roomName'] not in database:
        sio.emit('similarityError', {"msg": "Room not initialized"})
        return
    room = database[data['roomName']]
    room['ended'] = True
    print('end chat')
    print(room)


@sio.event
def similarityQuery(sid, data):
    if data['roomName'] not in database:
        sio.emit('similarityError', {"msg": "Room not initialized"})
        return
    room = database[data['roomName']]
    if room['sectionIndex'] == 0 or room.get('ended', False):  # Introductions or question generation
        return
    print(room)

    #ret = st.similarity_query(data['text'], room['topic'], room['section'], verbose=False)
    ret = st.similarity_query(data['text'], room['topic'], room['sectionIndex'], num_results=NUM_RESULTS_SAME_TOPIC, verbose=False)
    print(ret)
    return ret


# --------------------------------


@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)


if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 3002)), app)
