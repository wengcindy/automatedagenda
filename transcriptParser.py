import json
import csv

VERBOSE = True
IGNORE_EMPTY_MESSAGES = True
IMPORT_LABELS = True
IMPORT_PROS_CONS = True
filename = "20190430-finance"
old_CSV_path = "."


def read(filename):
	with open(filename + '.json', 'r') as myfile:
		transcript = myfile.read()

	# parse json
	obj = json.loads(transcript)

	out = open(filename + '.csv', 'w')
	csvWriter = csv.writer(out)
	csvWriter.writerow(["username", "text", "start time", "end time","total time", "session", "section"])

	for session_name, session_obj in obj.items():
		sectionDict = get_section_times(session_obj['sectionData'])

		# TODO: Import labels, pros/cons

		# TODO: Create transcript CSV

		for audio in range(len(session_obj['audioData'])):
			for j in range(len(session_obj['audioData'][audio]['data'])):
				# print(obj['2019winter2']['audioData'][audio]['data'][j]['text'])
				# print("\n")
				csvWriter.writerow([session_obj['audioData'][audio]['username'], \
					session_obj['audioData'][audio]['data'][j]['text'], \
					session_obj['audioData'][audio]['startTime'], \
					session_obj['audioData'][audio]['endTime'], \
					session_obj['audioData'][audio]['endTime']/1000 - session_obj['audioData'][audio]['startTime']/1000, \
					session_name, \
					match_section(sectionDict, session_obj['audioData'][audio]['startTime'], session_obj['audioData'][audio]['endTime'])])


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


if __name__ == "__main__":
	IMPORT_LABELS = (input("Do you want to input conversation labels manually? (Y/N):") == "Y")
	IMPORT_PROS_CONS = (input("Do you want to input true sentiment (pros/cons) labels manually? (Y/N):") == "Y")

	read(filename)





