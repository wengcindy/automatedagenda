import json
import csv

def read(filename):
	with open(filename + '.json', 'r') as myfile:
	    transcript=myfile.read()

	# parse json
	obj = json.loads(transcript)

	out = open(filename + '.csv', 'w')
	csvWriter = csv.writer(out)
	csvWriter.writerow(["username", "text", "start time", "end time","total time", "session", "section"])

	# list of all sessions
	sessions = []
	for session in obj:
		sessions.append(session)

	# get stats from each session
	for session in obj:
		sectionDict = makeDictionary(obj, session)
		for audio in range(len(obj[session]['audioData'])):
			for j in range(len(obj[session]['audioData'][audio]['data'])):
				# print(obj['2019winter2']['audioData'][audio]['data'][j]['text'])
				# print("\n")
				csvWriter.writerow([obj[session]['audioData'][audio]['username'], \
					obj[session]['audioData'][audio]['data'][j]['text'], \
					obj[session]['audioData'][audio]['startTime'], \
					obj[session]['audioData'][audio]['endTime'], \
					obj[session]['audioData'][audio]['endTime']/1000 - obj[session]['audioData'][audio]['startTime']/1000, \
					session, \
					matchSection(sectionDict, obj[session]['audioData'][audio]['startTime'], obj[session]['audioData'][audio]['endTime'])])

# create dictionary for each session
def makeDictionary(obj, session):
	# key: section name (A1, A2, ...), value: array with start and end times
	sectionDict = dict()

	for section in range(len(obj[session]['sectionData'])):
		if(obj[session]['sectionData'][section]['name'].startswith('A')):
			# append the time to the existing array for the section
			if obj[session]['sectionData'][section]['name'] in sectionDict:
				sectionDict[obj[session]['sectionData'][section]['name']].append(obj[session]['sectionData'][section]['startTime'])
				sectionDict[obj[session]['sectionData'][section]['name']].append(obj[session]['sectionData'][section]['endTime'])
			# create a new array for this section
			else:
				sectionList = []
				sectionList.append(obj[session]['sectionData'][section]['startTime'])
				sectionList.append(obj[session]['sectionData'][section]['endTime'])
				sectionDict[obj[session]['sectionData'][section]['name']] = sectionList
	return sectionDict

# match user's start/end time to a section
def matchSection(sectionDict, start, end):
	for i in sectionDict:
		if start >= sectionDict[i][0] and end <= sectionDict[i][1]:
			return i

filename = '20190430-finance'
read(filename)





