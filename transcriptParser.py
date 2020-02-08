import json
import csv
import math
import requests

import pandas as pd
import os
from nltk.tokenize import sent_tokenize
import numpy as np
import matplotlib.pyplot as plt

from similarityTester import *
from dataAnalyzer import DataAnalyzer


"""
This module reads data from the 10 transcripts available from February to
April 2019.
It classifies each sentence in the transcript into pros or cons (or positive/
negative), then write these labels as well as the cumulative data of each
agenda item into CSV files. It also records the true labels of each sentence
and the cumulative data either from manual input or imported from previous CSV 
files.

For classification of *sentences*, either of these two approaches can be used:
- Using the text similarity model in the similarityTester module. This gives
  the top pros/cons arguments in the agenda that's most similar to the current 
  sentence, and then classify the sentence based on whether these "top matches"
  are pros or cons.
- Using the MeaningCloud sentiment analysis API. This gives the tone of the
  sentence as positive or negative, and then use that to approximate whether
  it's about pros or cons.
You can choose the approach by changing the MODE variable.

For the Text Similarity approach, this module also performs cross-validation
on the similarity threshold for a match to contribute to classification
(i.e. matches with similarity score below that will be considered as neutral),
as well as the number of matches within the same section to take into account.
Earlier tests suggest an optimal value of 0.3 as threshold and using only 1
top match within the section.

The following output files are produced:
- [room name]_transcript_[mode].csv: Predicted and (manually labeled) true 
  label of each sentence as a pro or con.
  - For Text Similarity, also outputs the most similar pro/con point in 3
    different scopes: Within the entire database, within the current topic
    (e.g. immigration), and within the current section (agenda item) e.g. 
    immigration A3.
    Also records the actual pro/con point matched to the sentence (manually
    labeled), either by user input or imported from previous CSV files.
- [room name]_[mode].csv: Statistics of conversation for each section, 
  including features that we believe will be useful for classifying the
  conversation as a whole. Specifically, for each section/agenda item, it
  keeps track of these rubrics *after each participant's speech*:
  - Number of people spoken since the start of this section 
    (can be repeated, e.g. if the same person speaks twice)
  - Total speaking time since the start of this section
  - Number of people spoken about pros since the start of this section 
  	(Weighted by the extent for Sentiment Analysis, i.e. positive vs strongly 
  	positive)
  - Total speaking time about pros since the start of this section 
  	(Weighted by the extent for Sentiment Analysis)
  - Number of people spoken about cons since the start of this section 
  	(Weighted by the extent for Sentiment Analysis)
  - Total speaking time about cons since the start of this section 
  	(Weighted by the extent for Sentiment Analysis)
  - Number of people spoken weighted by their sentiment/pro/con scores since 
    the start of this section - this is realistically (# pros) - (# cons)
  - Total speaking time weighted by their sentiment/pro/con scores since 
    the start of this section - this is realistically (# pros) - (# cons)
  - TRUE label of current status: Whether the discussion up to this point is:
    - Insufficient in both pros and cons (0)
    - Sufficient in pros but not cons (1)
    - Sufficient in cons but not pros (2)
    - Sufficient in both pros and cons (3)
- An image file showing the Confusion Matrix of the sentence classification
  model on all data from all transcripts.
- In the case of text similarity, a plot of accuracy vs neutral threshold.
    
HOW TO USE:
1. Make sure all the following files are present in the same directory:
  - all_agendas.json (Agendas of all topics with pros and cons arguments,
    can be generated from running the web app)
  - 2019winter.json, 2019baylor.json, 20190430-finance.json (Transcript files,
    obtained from https://stanforddeliberate.org/viz/2019-2-19/data.js)
    - https://stanforddeliberate.org/viz/2019-2-19
    - https://stanforddeliberate.org/viz/2019-2-26
    - https://stanforddeliberate.org/viz/2019-3-21
    - https://stanforddeliberate.org/viz/2019-4-30
  - similarityTester.py
  - dataAnalyzer.py (For accuracy and confusion matrix plots)
2. Choose the model you want to use (Text Similarity/Sentiment Analysis) by
  changing the MODE variable. You can also adjust other parameters such as 
  cutoffs_to_test and best_matches_count_test.
3. Run this Python file.
4. You'll be prompted to choose whether you want to manually input the 
  conversation labels (0/1/2/3) and the transcript labels. 
  If you want to relabel them, enter Y.
  Otherwise, enter N and the program will import the labels from the existing
  CSV files if they exist. If importing fails, you will still have to input
  labels manually.

Note: 
- BACKUP THE CSV FILES before you run the script as much as possible, 
  especially if you've made changes to the code that might cause bugs.
  If not, in the event of runtime errors (which hopefully shouldn't happen
  unless the code is changed), some manually labeled data might be LOST. 
  (This is due to the program overwriting the old CSV file. There should be
  a way to fix it.)
- The MeaningCloud Sentiment Analysis API has a monthly quota of 20,000 
  requests. There's also a limit of 2 requests per second, so running this
  script might take a moderate amount of time.
- Currently a user's entire speech is processed at once. But it's possible to
  analyze each sentence separately (TBC) 
"""


MODE = 'Sentiment Analysis'  # ['Text Similarity', 'Sentiment Analysis']

VERBOSE = False
PRINT_MESSAGES = False
IGNORE_EMPTY_MESSAGES = True
IMPORT_LABELS = True
IMPORT_PROS_CONS = True

NEUTRAL_CUTOFF = 0.1  # placeholder, replaced by values in cutoffs_to_test at runtime
BEST_MATCHES_COUNT = 1  # placeholder, replaced by values in best_matches_count_test at runtime
best_matches_count_test = [1] #range(1, 5)
cutoffs_to_test = np.arange(0, 0.8, 0.1)

filename = ["2019winter", "2019baylor", "20190430-finance"]  # "climate" for Sentiment Analysis only
topics_dict = {"2019winter": "immigration", "2019baylor": "immigration", "20190430-finance": "campaignFinanceReform"}
topic = None
old_CSV_path = "."
headers = ["Session",
		   "Section",
		   "Speech ID",
		   "Username",
		   "Text",
		   "Start time",
		   "End time",
		   "Individual speaking time",
		   "Individual sentiment",
		   "# people spoken",
		   "Total speaking time",
		   "Pros # ppl spoken",
		   "Pros speaking time",
		   "Cons # ppl spoken",
		   "Cons speaking time",
		   "# ppl spoken (weighted)",
		   "Total speaking time (weighted)",
		   "Label of current status"]
transcript_headers = ["Section ID",
					  "Speech ID",
					  "Sentence ID",
					  "User Name",
					  "Text",
					  "Predicted pros/cons label",
					  "Best match",
					  "Similarity Best match",
					  "Best match in topic",
					  "Similarity Best match in topic",
					  "Best match in section",
					  "Similarity Best match in section",
					  "True sentiment",
					  "True match"] if MODE == 'Text Similarity' else [
					  "Section ID",
					  "Speech ID",
					  "Sentence ID",
					  "User Name",
					  "Text",
					  "Predicted sentiment",
					  "True sentiment"]

old_labels = {}
old_sentiments = {}

data_analyzer = DataAnalyzer()


class StatsRecorder:
	def __init__(self, session, csv_writer, transcript_writer):
		self.session = session
		self.csv_writer = csv_writer
		self.transcript_writer = transcript_writer
		self.section = ''
		if MODE == 'Text Similarity':
			self.st = SimilarityTester()

		# For current speech
		self.username = None
		self.audio_id = 0
		self.start_time = 0
		self.end_time = 0
		self.speech_time = 0
		self.weighted_sentiment = 0
		self.label = 0
		self.sentences = []
		self.predicted_sentiments = []
		self.true_sentiments = []
		if MODE == 'Text Similarity':
			self.best_matches = []
			self.best_matches_similarities = []
			self.best_matches_same_topic = []
			self.best_matches_same_topic_similarities = []
			self.best_matches_same_section = []
			self.best_matches_same_section_similarities = []
			self.true_matches = []

		# For the entire section
		self.section_id = 0
		self.num_spoken = 0
		self.time_spoken = 0
		self.num_spoken_pros = 0
		self.time_spoken_pros = 0
		self.num_spoken_cons = 0
		self.time_spoken_cons = 0
		self.num_spoken_weighted = 0
		self.time_spoken_weighted = 0

	def new_section(self, section_name):
		self.section = section_name

		# For current speech
		self.username = None
		self.audio_id = 0
		self.start_time = 0
		self.end_time = 0
		self.speech_time = 0
		self.weighted_sentiment = 0
		self.label = 0
		self.sentences = []
		self.predicted_sentiments = []
		self.true_sentiments = []
		if MODE == 'Text Similarity':
			self.best_matches = []
			self.best_matches_similarities = []
			self.best_matches_same_topic = []
			self.best_matches_same_topic_similarities = []
			self.best_matches_same_section = []
			self.best_matches_same_section_similarities = []
			self.true_matches = []

		# For the entire section
		self.section_id = 0
		self.num_spoken = 0
		self.time_spoken = 0
		self.num_spoken_pros = 0
		self.time_spoken_pros = 0
		self.num_spoken_cons = 0
		self.time_spoken_cons = 0
		self.num_spoken_weighted = 0
		self.time_spoken_weighted = 0

	def add_speech(self, id, user, speech, start_time, end_time):
		"""
		Add a new speech, i.e. an entire audio of one person speaking.

        Break down the speech into sentences and analyze the pro/cons of each.

        At the end, obtains the label either by user input or importing.

		:param id: ID of speech from JSON (for importing labels)
		:param user: Name of current speaker
		:param speech: Text of entire speech
		:param start_time: Start time of this speech, as long nubmer
		:param end_time: End time of this speech, as long nubmer
		:return:
		"""
		self.username = user
		self.audio_id = id
		self.start_time = start_time
		self.end_time = end_time
		self.speech_time = end_time/1000 - start_time/1000
		self.weighted_sentiment = 0
		self.predicted_sentiments = []
		if MODE == 'Text Similarity':
			self.best_matches = []
			self.best_matches_similarities = []
			self.best_matches_same_topic = []
			self.best_matches_same_topic_similarities = []
			self.best_matches_same_section = []
			self.best_matches_same_section_similarities = []

		self.num_spoken += 1
		self.time_spoken += self.speech_time

		#self.sentences = sent_tokenize(speech)
		#speech = ''.join(self.sentences)
		#for sentence in self.sentences:
		#	self.add_sentence(sentence, len(sentence) / len(speech))
		self.sentences = [speech]  # For compatibility
		self.add_sentence(speech, 1)  # Temporary fix

		self.label = self.read_label()
		self.true_sentiments, self.true_matches = self.read_true_sentiments()
		#input("Press enter to continue...")  # TODO: Debug
		#print()
		self.write_csv()
		self.write_transcript()

		for i in range(len(self.predicted_sentiments)):
			data_analyzer.add(self.predicted_sentiments[i], self.true_sentiments[i])

	def add_sentence(self, sentence, length_ratio):
		"""
		Add a new sentence. Increment corresponding counters.

        The increase is according to the proportion this sentence
        takes in the person's entire speech in terms of characters.
        For example, a 40-character sentence about pros in the
        entire speech of 100 characters will contribute 0.4 to
        the number of people spoken about pros.
		:param sentence: Sentence as string
		:param length_ratio: Proportion of this sentence's length to the
			total length of that person's speech
		:return: Score of this sentence (-1 to 1)
		"""
		if PRINT_MESSAGES:
			print(self.username + ":" + sentence)

		# TODO: Deal with sentiment analysis
		if MODE == 'Text Similarity':
			data = self.st.similarity_query(
				sentence, topic, int(self.section[1:]), neutral_cutoff=NEUTRAL_CUTOFF, best_matches_count=BEST_MATCHES_COUNT, verbose=VERBOSE)
			label = data['label']
			match = data['best_match_old']
			match_sim = data['best_match_similarity']
			match_topic = data['best_matches_same_topic_old'][0]
			match_topic_sim = data['best_matches_same_topic_similarity'][0]
			match_section = data['best_match_same_section_old']
			match_section_sim = data['best_match_same_section_similarity']
			labels = ["Con", "Neutral", "Pro"]
			score = labels.index(label) - 1
		elif MODE == 'Sentiment Analysis':
			API_ENDPOINT = "https://api.meaningcloud.com/sentiment-2.1"
			data = {
				'key':'7a4a4878fd419d831e44ea7ed1549149',
				'lang':'en',
				'txt': sentence
			}
			r = requests.post(url = API_ENDPOINT, data = data)
			results_json = r.json()
			while results_json['status']['msg'] != 'OK':
				if results_json['status']['msg'] == 'Request rate limit exceeded':
					r = requests.post(url = API_ENDPOINT, data = data)
					results_json = r.json()
				else:
					print('API error when running sentiment analysis: ' + results_json['status']['msg'])
					return
			score_tag = results_json['score_tag']
			score = -1 if score_tag in ['N+', 'N'] else 1 if score_tag in ['P+', 'P'] else 0

		self.predicted_sentiments.append(score)
		if MODE == 'Text Similarity':
			self.best_matches.append(match)
			self.best_matches_similarities.append(match_sim)
			self.best_matches_same_topic.append(match_topic)
			self.best_matches_same_topic_similarities.append(match_topic_sim)
			self.best_matches_same_section.append(match_section)
			self.best_matches_same_section_similarities.append(match_section_sim)
		self.weighted_sentiment += length_ratio * score
		self.num_spoken_weighted += length_ratio * score
		self.time_spoken_weighted += self.speech_time * length_ratio * score
		if score > 0:
			self.num_spoken_pros += length_ratio * score
			self.time_spoken_pros += self.speech_time * length_ratio * score
		else:
			self.num_spoken_cons -= length_ratio * score
			self.time_spoken_cons -= self.speech_time * length_ratio * score
		if VERBOSE:
			print("Pros/cons label: " + str(score))
		return score

	def read_label(self):
		"""
		Reads the label for current speech (status at the end of this),
		either from user input or from existing sources.
		"""
		if IMPORT_LABELS:
			try:
				return old_labels[self.session][self.audio_id]
			except KeyError:
				pass

		s = input("Please enter label: 1 for sufficient pros, 2 for sufficient cons, 3 or 12 for sufficient in both: ")
		if "3" in s:
			return 3
		else:
			return (1 if "1" in s else 0) + (2 if "2" in s else 0)

	def read_true_sentiments(self):
		"""
		Reads the true sentiment for current sentence, either from user
		input or from existing sources.
		:return: List of true sentiment scores (-1 or 1) for each sentence
		"""
		scores = []
		sentence_num = len(self.sentences)
		if IMPORT_PROS_CONS:
			try:
				#for i in range(sentence_num):
				#	scores.append(old_sentiments[self.session][self.audio_id][i])
				#return scores
				scores_old = [a for a,b in old_sentiments[self.session][self.audio_id].values()]
				matches = [b for a,b in old_sentiments[self.session][self.audio_id].values()]  # Should be '' if sentiment analysis (TODO: Check)

				if -0.2 <= sum(scores_old) * 1.0 / len(scores_old) <= 0.2:
					return [0], [matches[0]]
				else:
					return [1 if sum(scores_old) > 0 else -1], [matches[0]]
			except KeyError:
				pass

		s = ''
		while len(s) != sentence_num:
			s = input("Please enter %d pro/con scores: 1 for pro, 0 for con, space for neutral (e.g.1 010):" % sentence_num)
		for c in s:
			scores.append(1 if c == '1' else (-1 if c == '0' else 0))

		# TODO: Deal with sentiment analysis
		if MODE == 'Text Similarity':
			match = input("Please enter the actual pro/con point being discussed (format: \"pro 2\" or \"2\"):")
			if match and "pro" not in match and "con" not in match:
				match = "%s %s" % ("pro" if scores[0] == 1 else "con", match)
			match = ("%s %s %s" % (topic, self.section, match)).strip() if match else ''
			matches = [match] * sentence_num  # Temporary
			return scores, matches

	def write_csv(self):
		"""
		Write current statistics to the CSV output file.
		Called after each speech.
		"""
		self.csv_writer.writerow([
			self.session,
			self.section,
			self.audio_id,
			self.username,
			' '.join(self.sentences),
			self.start_time,
			self.end_time,
			self.speech_time,
			self.weighted_sentiment,
			self.num_spoken,
			self.time_spoken,
			self.num_spoken_pros,
			self.time_spoken_pros,
			self.num_spoken_cons,
			self.time_spoken_cons,
			self.num_spoken_weighted,
			self.time_spoken_weighted,
			self.label
		])

	def write_transcript(self):
		"""
		Write transcripts and sentiment scores to the CSV output file.
		Called after each speech, but adds all sentences in separate entries.
		"""
		for i in range(len(self.sentences)):
			self.transcript_writer.writerow([
				self.section,
				self.audio_id,
				i,
				self.username,
				'"' + self.sentences[i] + '"',
				self.predicted_sentiments[i],
				self.best_matches[i],
				self.best_matches_similarities[i],
				self.best_matches_same_topic[i],
				self.best_matches_same_topic_similarities[i],
				self.best_matches_same_section[i],
				self.best_matches_same_section_similarities[i],
				self.true_sentiments[i],
				self.true_matches[i]
			] if MODE == 'Text Similarity' else [
				self.section,
				self.audio_id,
				i,
				self.username,
				'"' + self.sentences[i] + '"',
				self.predicted_sentiments[i],
				self.true_sentiments[i]
			])


def read(filename):
	global topic
	topic = topics_dict[filename]

	with open(filename + '.json', 'r') as myfile:
		transcript = myfile.read()

	# parse json
	obj = json.loads(transcript)

	out = open(filename + '_' + MODE + '.csv', 'w', newline='')
	csv_writer = csv.writer(out)
	csv_writer.writerow(headers)

	if IMPORT_LABELS:
		initialize_labels(sessions=obj.keys())

	for session_name, session_obj in obj.items():
		data_analyzer.new_session(session_name)

		section_dict = get_section_times(session_obj['sectionData'])

		if IMPORT_PROS_CONS:
			initialize_transcripts(session_name)

		out2 = open(session_name + '_transcript_' + MODE + '.csv', 'w', newline='')
		transcript_writer = csv.writer(out2)
		transcript_writer.writerow(transcript_headers)

		current_section = None
		stats = StatsRecorder(session_name, csv_writer, transcript_writer)

		for speech in session_obj['audioData']:
			# Checks if a new section needs to be started, and whether
			# this speech should count at all
			new_section = match_section(section_dict, speech['startTime'],
										speech['endTime'])
			if new_section is None:
				continue
			if new_section != current_section:
				current_section = new_section
				stats.new_section(new_section)
				out.flush()
				out2.flush()
				if PRINT_MESSAGES:
					print("---------")
					print("New section")
					print("---------")

			process_speech(speech, stats)
			"""for j in range(len(speech['data'])):
				# print(obj['2019winter2']['audioData'][audio]['data'][j]['text'])
				# print("\n")
				csv_writer.writerow([speech['username'], \
					speech['data'][j]['text'], \
					speech['startTime'], \
					speech['endTime'], \
					speech['endTime']/1000 - speech['startTime']/1000, \
					session_name, \
					match_section(section_dict, speech['startTime'], speech['endTime'])])"""

		out2.close()

	out.close()


def process_speech(speech, stats):
	"""
	Process a single speech, i.e. an audio of one person speaking.
    Add information of each sentence in the speech.
	:param speech: Speech data as JSON object
	:param stats: Stats recorder with ongoing records from the session
	"""
	id = speech['id']
	username = speech['username'].replace(',', '')  # Prevent messing up with CSV

	# Each audio might be broken down into several sentences.
	# Combine all sentences
	speech_text = ''.join([sentence['text'] for sentence in speech['data']])
	if len(speech_text) == 0 and IGNORE_EMPTY_MESSAGES:
		return

	# Remove adverb "like"'s to prevent messing up sentiments
	speech_text.replace(' like ', '')

	stats.add_speech(id, username, speech_text,
					 speech['startTime'], speech['endTime'])


def get_section_times(section_data):
	"""
	Get the time at which each section starts/ends from the JSON transcript.
	:param section_data: "sectionData" JSON array from the transcript
	:return: Dict mapping section names that start with A (A1, A2, ...)
		to list of start and end times (can have multiple pairs of them)
	"""
	section_dict = dict()

	for section in section_data:
		section_name = section['name']
		section_list = [section['startTime'], section['endTime']]
		if (section_name.startswith('A')):
			# append the time to the existing array for the section
			if section_name in section_dict:
				section_dict[section_name] += section_list
			# create a new array for this section
			else:
				section_dict[section_name] = section_list
	return section_dict


def match_section(section_dict, start, end):
	"""
	Find the section of a user's speech.
	:param section_dict: Dict mapping section names to start and end times
	:param start: Start time of user's speech
	:param end: End time of user's speech
	:return: Section name, or None if it's not in any section
	"""
	for section_name, times in section_dict.items():
		for j in range(int(len(times) / 2)):
			if start >= times[j*2] and end <= times[j*2+1]:
				return section_name
	return None


def initialize_labels(sessions=None):
	"""
	Read labels from past CSVs, if applicable.
	Assumes CSV file is stored in oldCSVPath and named as the file name.
	"""
	session_header = "Session"
	speech_id_header = "Speech ID"
	label_header = "Label of current status"

	try:
		df = pd.read_csv(os.path.join(old_CSV_path, filename + ".csv"))
		for i in range(len(df[session_header])):
			session = df[session_header][i]
			sessions.add(session)
			speech_id = df[speech_id_header][i]
			label = df[label_header][i]
			if session not in old_labels:
				old_labels[session] = {}
			old_labels[session][speech_id] = label
	except Exception:
		pass

	# Read labels from each session's CSV for compatibility
	if not sessions:
		return
	for session in sessions:
		try:
			try:
				df = pd.read_csv(os.path.join(old_CSV_path, session + "_" + MODE + ".csv"))
			except FileNotFoundError:
				df = pd.read_csv(os.path.join(old_CSV_path, session + ".csv"))
			for i in range(len(df[speech_id_header])):
				speech_id = df[speech_id_header][i]
				label = df[label_header][i]
				if session not in old_labels:
					old_labels[session] = {}
				if speech_id not in old_labels[session]:  # Prevent from overridding new data
					old_labels[session][speech_id] = label
		except Exception:
			pass


def initialize_transcripts(session):
	"""
	Read true sentiments from past CSVs, if applicable.
	Assumes CSV file is stored in oldCSVPath and named as the session
	name concatenated with "_transcript.csv".
	:param session: Name of session (room)
	"""
	speech_id_header = "Speech ID"
	sentence_id_header = "Sentence ID"
	label_header = "True sentiment"
	match_header = "True match"

	if session not in old_sentiments:
		old_sentiments[session] = {}
	try:
		try:
			df = pd.read_csv(os.path.join(old_CSV_path, session + "_transcript_" + MODE + ".csv"))
		except FileNotFoundError:
			df = pd.read_csv(os.path.join(old_CSV_path, session + "_transcript.csv"))
		for i in range(len(df[speech_id_header])):
			speech_id = df[speech_id_header][i]
			sentence_id = df[sentence_id_header][i]
			label = df[label_header][i]
			match = df[match_header][i] if match_header in df else ''
			if not match or match == 'nan' or (type(match) is float and math.isnan(match)):
				match = ''
			if session not in old_sentiments:
				old_sentiments[session] = {}
			if speech_id not in old_sentiments[session]:
				old_sentiments[session][speech_id] = {}
			old_sentiments[session][speech_id][sentence_id] = (label, match)
	except Exception:
		pass


if __name__ == "__main__":
	IMPORT_LABELS = (input("Do you want to input conversation labels manually? (Y/N):") != "Y")
	IMPORT_PROS_CONS = (input("Do you want to input true sentiment (pros/cons) labels manually? (Y/N):") != "Y")

	# parse agenda file
	with open("all_agendas.json", 'r') as myfile:
		data = myfile.read()
	data = json.loads(data)
	initialize(data['data'])

	data_analyzer = DataAnalyzer()
	best_accuracy = 0
	best_cutoff = 0
	best_count = 0

	if MODE == 'Text Similarity':  # Cross-validation needed
		accuracy_per_cutoff = []
		for cutoff in cutoffs_to_test:
			for best_matches_count in best_matches_count_test:
				NEUTRAL_CUTOFF = cutoff
				BEST_MATCHES_COUNT = best_matches_count

				data_analyzer.reset()
				for file in filename:
					read(file)
				print("Cutoff = %f, matches = %d" % (cutoff, best_matches_count))

				data_analyzer.print_session_accuracies()
				accuracy = data_analyzer.get_accuracy()
				print("Overall accuracy: %f" % accuracy)
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_cutoff = cutoff
					best_count = best_matches_count
				accuracy_per_cutoff.append(accuracy)

		print("Best accuracy: %f, Cutoff: %f, # matches: %d" % (best_accuracy, best_cutoff, best_count))
		NEUTRAL_CUTOFF = best_cutoff
		BEST_MATCHES_COUNT = best_count
		data_analyzer.reset()
		for file in filename:
			read(file)  # Repopulate CSV files
		data_analyzer.print_session_accuracies()
		data_analyzer.plot_confusion_matrix(
			title="Sentence pro/con classifications using " + MODE,
			filename="%s, Cutoff=%f, Matches=%d, Accuracy=%f" % (MODE, best_cutoff, best_count, best_accuracy),
			normalize=True)

		plt.clf()
		fig, ax = plt.subplots()
		ax.plot(cutoffs_to_test, accuracy_per_cutoff)
		ax.set(xlabel='Similarity threshold',
			   ylabel='Accuracy',
			   title='Determining similarity threshold for sentence to be classified as neutral')
		plt.savefig('Text Similarity_Neutral threshold.png')

	else:
		data_analyzer.reset()
		for file in filename:
			read(file)

		data_analyzer.print_session_accuracies()
		accuracy = data_analyzer.get_accuracy()
		print("Overall accuracy: %f" % accuracy)
		data_analyzer.plot_confusion_matrix(
			title="Sentence pro/con classifications using " + MODE,
			filename="%s, Accuracy=%f" % (MODE, accuracy),
			normalize=True)
