import json
import csv
import pandas as pd
from Lib import os

VERBOSE = True
IGNORE_EMPTY_MESSAGES = True
IMPORT_LABELS = True
IMPORT_PROS_CONS = True
filename = "20190430-finance"
old_CSV_path = "."
headers = ["username", "text", "start time", "end time","total time", "Session", "section"]
transcript_headers = ["Section ID",
					  "Speech ID",
					  "Sentence ID",
					  "User Name",
					  "Text",
					  "Predicted pros/cons label",
					  "True sentiment"]

old_labels = {}
old_sentiments = {}

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
		for speech in session_obj['audioData']:
			# Checks if a new section needs to be started, and whether
			# this speech should count at all
			new_section = match_section(section_dict, speech['startTime'],
										speech['endTime'])
			if new_section is None:
				continue
			if new_section != current_section:
				current_section = new_section
				# stats.newSection();
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





