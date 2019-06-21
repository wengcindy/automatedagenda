import json
import csv
import pandas as pd
from Lib import os
from nltk.tokenize import sent_tokenize

VERBOSE = True
IGNORE_EMPTY_MESSAGES = True
IMPORT_LABELS = True
IMPORT_PROS_CONS = True
filename = "20190430-finance"
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
					  "True sentiment"]

old_labels = {}
old_sentiments = {}


class StatsRecorder:
	def __init__(self, session, csv_writer, transcript_writer):
		self.session = session
		self.csv_writer = csv_writer
		self.transcript_writer = transcript_writer
		self.section = ''

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

		self.csv_writer.flush()
		self.transcript_writer.flush()

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

		self.num_spoken += 1
		self.time_spoken += self.speech_time

		self.sentences = sent_tokenize(speech)
		speech = ''.join(self.sentences)
		for sentence in self.sentences:
			#self.add_sentence(sentence, len(sentence) / len(speech))
			pass

		self.label = self.read_label()
		self.true_sentiments = self.read_true_sentiments()
		self.write_csv()
		self.write_transcript()

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
		print(self.username + ":" + sentence)
		score = 0  # USE gensim2 HERE
		self.predicted_sentiments.append(score)
		self.weighted_sentiment += length_ratio * score
		self.num_spoken_weighted += length_ratio * score
		self.time_spoken_weighted += self.speech_time * length_ratio * score
		if score > 0:
			self.num_spoken_pros += length_ratio * score
			self.time_spoken_pros += self.speech_time * length_ratio * score
		else:
			self.num_spoken_cons += length_ratio * score
			self.time_spoken_cons += self.speech_time * length_ratio * score
		if VERBOSE:
			print("Pros/cons label: " + score)
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
				for i in range(sentence_num):
					scores.append(old_sentiments[self.session][self.audio_id][i])
				return scores
			except KeyError:
				pass

		s = ''
		while len(s) != sentence_num:
			s = input("Please enter %d sentiment scores: 1 for positive, 0 for negative, space for neutral (e.g.1 010):" % sentence_num)
		for c in s:
			scores.append(1 if c == '1' else (-1 if c == '0' else 0))
		return scores

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
				self.true_sentiments[i]
			])


def read(filename):
	with open(filename + '.json', 'r') as myfile:
		transcript = myfile.read()

	# parse json
	obj = json.loads(transcript)

	out = open(filename + '.csv', 'w')
	csv_writer = csv.writer(out)
	csv_writer.writerow(headers)

	if IMPORT_LABELS:
		initialize_labels()

	for session_name, session_obj in obj.items():
		section_dict = get_section_times(session_obj['sectionData'])

		if IMPORT_PROS_CONS:
			initialize_transcripts(session_name)

		# TODO: Create transcript CSV
		out2 = open(session_name + '_transcript.csv', 'w')
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
				print("---------")
				print("New section")
				print("---------")

			# processSpeech(singleSpeech, stats);
			for j in range(len(speech['data'])):
				# print(obj['2019winter2']['audioData'][audio]['data'][j]['text'])
				# print("\n")
				csv_writer.writerow([speech['username'], \
					speech['data'][j]['text'], \
					speech['startTime'], \
					speech['endTime'], \
					speech['endTime']/1000 - speech['startTime']/1000, \
					session_name, \
					match_section(section_dict, speech['startTime'], speech['endTime'])])

		out2.close()

	out.close()


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


def initialize_labels():
	"""
	Read labels from past CSVs, if applicable.
	Assumes CSV file is stored in oldCSVPath and named as the file name.
	"""
	session_header = "Session"
	speech_id_header = "Speech ID"
	label_header = "Label of current status"

	df = pd.read_csv(os.path.join(old_CSV_path, filename + ".csv"))
	for i in range(len(df[session_header])):
		session = df[session_header][i]
		speech_id = df[speech_id_header][i]
		label = df[label_header][i]
		if session not in old_labels:
			old_labels[session] = {}
		old_labels[session][speech_id] = label


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

	df = pd.read_csv(os.path.join(old_CSV_path, session + "_transcript.csv"))
	for i in range(len(df[speech_id_header])):
		speech_id = df[speech_id_header][i]
		sentence_id = df[sentence_id_header][i]
		label = df[label_header][i]
		if session not in old_labels:
			old_labels[session] = {}
		if speech_id not in old_labels[session]:
			old_labels[session][speech_id] = {}
		old_labels[session][speech_id][sentence_id] = label


if __name__ == "__main__":
	IMPORT_LABELS = (input("Do you want to input conversation labels manually? (Y/N):") == "Y")
	IMPORT_PROS_CONS = (input("Do you want to input true sentiment (pros/cons) labels manually? (Y/N):") == "Y")

	read(filename)





