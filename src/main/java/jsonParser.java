import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import com.opencsv.CSVReader;
import kong.unirest.HttpResponse;
import kong.unirest.JsonNode;
import kong.unirest.Unirest;
import kong.unirest.UnirestException;
import org.json.*;
  
public class jsonParser {
    final static boolean VERBOSE = true;
    final static boolean IGNORE_EMPTY_MESSAGES = true;
    static boolean IMPORT_LABELS = true;
    static boolean IMPORT_PROS_CONS = true;
    final static String path = "climate.json";
    final static String oldCSVPath = ".";

    static int credits = 0;
    static Map<String, Map<Integer, Integer>> oldLabels = new HashMap<>();
    static Map<String, Map<Integer, Map<Integer, Integer>>> oldSentiments = new HashMap<>();

    static List<String> headers = Arrays.asList(
            "Section ID",
            "Speech ID",
            "Name",
            "Individual speaking time",
            "Individual sentiment",
            "# people spoken",
            "Total speaking time",
            //"Total number of people spoken about pros",
            //"Total speaking time about pros",
            //"Total number of people spoken about cons",
            //"Total speaking time about cons",
            "# ppl spoken (weighted)",
            "Total speaking time (weighted)",
            "Pros # ppl spoken (weighted)",
            "Pros speaking time (weighted)",
            "Cons # ppl spoken (weighted)",
            "Cons speaking time (weighted)",
            "Label of current status"
    );
    static List<String> transcriptHeaders = Arrays.asList(
            "Section ID",
            "Speech ID",
            "Sentence ID",
            "Name",
            "Text",
            "MeaningCloud sentiment",
            "True sentiment"
    );

    /**
     * Tool for analysis of sentiment and similarities to predetermined
     * pros/cons points.
     *
     * As of now, this class will generate an aggregate score used to
     * classify the message as discussing pros or cons of the agenda.
     */
    public static class SentimentAnalyzer {
        String message;
        boolean isEmpty;  // If the message is empty

        String sentimentTag;  // Sentiment (P+, P, NEU, N, N+, NONE)
        int sentimentScore;  // -2 (N+) to 2 (P+)
        boolean isAgreement;  // Agreement/disagreement tags from sentiment analysis (highly inaccurate)
        //boolean isSubjective;  // Subjective/objective tags from sentiment analysis
        int sentimentConfidence;  // int 0-100

        List<SentimentAnalyzer> sentences;  // For individual sentences

        public SentimentAnalyzer(String msg) {
            initMessage(msg);
        }

        public SentimentAnalyzer(JSONObject result) {
            processSentimentResult(result);
        }

        public void initMessage(String msg) {
            message = msg;
            // Check if the message is empty (has no letters, probably due to mic failure)
            if (msg.replaceAll("[^a-zA-Z0-9]","").equals("")) { // https://stackoverflow.com/questions/4945695/how-to-filter-string-for-unwanted-characters-using-regex
                isEmpty = true;
                return;
            }

            JSONObject result = getSentimentResult();
            if (result == null) {
                throw new NullPointerException("Program terminated due to fatal error with sentiment analysis.");
            }
            processSentimentResult(result);
        }

        private JSONObject getSentimentResult() {
            HttpResponse<JsonNode> response = null;
            JSONObject result = null;
            while (result == null) {
                try {
                    response = Unirest.post("https://api.meaningcloud.com/sentiment-2.1")
                            .header("content-type", "application/x-www-form-urlencoded")
                            .queryString("key", "7a4a4878fd419d831e44ea7ed1549149")
                            .field("lang", "en")
                            .field("txt", message)
                            //.body("key=7a4a4878fd419d831e44ea7ed1549149&lang=en&txt=" + message)
                            .asJson();
                } catch (UnirestException e) {
                    e.printStackTrace();
                    return null;
                }
                if (!response.isSuccess()) {
                    System.out.println("Error " + response.getStatus() + " when running Sentiment Analysis: " + response.getStatusText());
                    return null;
                }
                result = response.getBody().getObject();
                String msg = result.getJSONObject("status").getString("msg");
                if (!msg.equals("OK")) {
                    if (msg.equals("Request rate limit exceeded")) {
                        result = null;
                    } else {
                        System.out.println("API error when running Sentiment Analysis: " + msg);
                        return null;
                    }
                } else {
                    credits = result.getJSONObject("status").getInt("remaining_credits");
                }
            }
            return result;
        }

        private void processSentimentResult(JSONObject result) {
            //System.out.println(result);
            if (result.has("text")) {  // Sentence objects have text
                message = result.getString("text");
            }
            sentimentTag = result.getString("score_tag");
            // Convert letter score to numeric score
            List<String> sentiments = Arrays.asList("N+", "N", "NEU", "P", "P+");
            sentimentScore = sentiments.indexOf(sentimentTag) - 2;
            if (sentimentScore == -3) sentimentScore = 0;  // NONE
            isAgreement = result.getString("agreement").equals("AGREEMENT");
            /*if (result.has("subjectivity")) {  // Sentences have no subjectivity
                isSubjective = result.getString("subjectivity").equals("SUBJECTIVE");
            }*/
            sentimentConfidence = result.getInt("confidence");

            sentences = new ArrayList<>();
            if (result.has("sentence_list")) {
                JSONArray sentenceObjects = result.getJSONArray("sentence_list");
                for (int i=0; i<sentenceObjects.length(); i++) {
                    sentences.add(new SentimentAnalyzer(sentenceObjects.getJSONObject(i)));
                }
            }
        }
    }

    /**
     * Class that encapsulates all statistics of the ongoing conversation,
     * including the following data:
     *
     * - # of people spoken
     * - Total speaking time
     * - # of people spoken about pros
     * - Total speaking time about pros
     * - # of people spoken about cons
     * - Total speaking time about cons
     * - # of people spoken, weighted by sentiment score (-2 ~ 2)
     * - Total speaking time, weighted by sentiment score (-2 ~ 2)
     * - # of people spoken about pros, weighted by sentiment score (1 ~ 2)
     * - Total speaking time about pros, weighted by sentiment score (1 ~ 2)
     * - # of people spoken about cons, weighted by sentiment score (1 ~ 2)
     * - Total speaking time about cons, weighted by sentiment score (1 ~ 2)
     *
     * All stats are reset at the start of each topic/section.
     */
    public static class StatsRecorder {
        String sessionName = null;

        // For current speech
        String username = null;
        int audioId = 0;
        double speechTime = 0;
        double weightedSentiment = 0;
        int label = 0;
        List<String> sentences = new ArrayList<>();
        List<Double> predictedSentiments = new ArrayList<>();
        List<Integer> trueSentiments = new ArrayList<>();

        // For the entire section
        int sectionId = 0;
        int numSpoken = 0;
        double timeSpoken = 0;
        //double numSpokenPros = 0, timeSpokenPros = 0;
        //double numSpokenCons = 0, timeSpokenCons = 0;
        double numSpokenWeighted = 0, timeSpokenWeighted = 0;
        double numSpokenProsWeighted = 0, timeSpokenProsWeighted = 0;
        double numSpokenConsWeighted = 0, timeSpokenConsWeighted = 0;
        FileWriter csvWriter = null;
        FileWriter transcriptWriter = null;

        public StatsRecorder(String session, FileWriter csvWriter, FileWriter transcriptWriter) throws IOException {
            sessionName = session;
            this.csvWriter = csvWriter;
            this.transcriptWriter = transcriptWriter;
            sectionId = -1;  // incremented to 0 in newSection
            newSection();
        }

        public void newSection() throws IOException {
            sectionId += 1;
            username = null;
            audioId = 0;
            speechTime = 0;
            weightedSentiment = 0;
            numSpoken = 0;
            timeSpoken = 0;
            //numSpokenPros = 0; timeSpokenPros = 0;
            //numSpokenCons = 0; timeSpokenCons = 0;
            numSpokenWeighted = 0; timeSpokenWeighted = 0;
            numSpokenProsWeighted = 0; timeSpokenProsWeighted = 0;
            numSpokenConsWeighted = 0; timeSpokenConsWeighted = 0;
            csvWriter.flush();
            transcriptWriter.flush();
        }

        /**
         * Add a new speech, i.e. an entire audio of one person speaking.
         *
         * Use MeaningCloud API to break down into sentences and analyze
         * the sentiments of each.
         *
         * At the end, obtains the label either by user input or importing.
         *
         * @param id ID of speech from JSON (for importing labels)
         * @param user Name of current speaker
         * @param speech Text of entire speech
         * @param startTime Start time of this speech, as long nubmer
         * @param endTime End time of this speech, as long nubmer
         */
        public void addSpeech(int id, String user, String speech, long startTime, long endTime) throws IOException {
            username = user;
            audioId = id;
            speechTime = ((double) endTime - (double) startTime) / 1000;
            weightedSentiment = 0;
            numSpoken++;
            timeSpoken += speechTime;
            int length = speech.length();

            sentences = new ArrayList<>();
            predictedSentiments = new ArrayList<>();
            SentimentAnalyzer sa = new SentimentAnalyzer(speech);
            if (sa.sentences.isEmpty()) {
                addSentence(sa, 1.0);
            } else {
                for (SentimentAnalyzer sentence : sa.sentences) {
                    addSentence(sentence, sentence.message.length() * 1.0 / length);
                }
            }

            label = readLabel();
            trueSentiments = readTrueSentiment();
            writeCSV();
            writeTranscript();
        }

        /**
         * Add a new sentence. Increment corresponding counters.
         *
         * The increase is according to the proportion this sentence
         * takes in the person's entire speech in terms of characters.
         * For example, a 40-character sentence about pros in the
         * entire speech of 100 characters will contribute 0.4 to
         * the number of people spoken about pros.
         * @param sa SentimentAnalyzer object of the sentence
         * @param lengthRatio Proportion of this sentence's length to the
         *                    total length of that person's speech
         * @return Score of this sentence
         */
        public double addSentence(SentimentAnalyzer sa, double lengthRatio) throws IOException {
            sentences.add(sa.message);
            System.out.println(username + ": " + sa.message);
            double score = sa.sentimentScore;
            predictedSentiments.add(score);
            weightedSentiment += lengthRatio * score;
            numSpokenWeighted += lengthRatio * score;
            timeSpokenWeighted += speechTime * lengthRatio * score;
            if (score > 0) {
                //numSpokenPros += lengthRatio;
                //timeSpokenPros += speechTime * lengthRatio;
                numSpokenProsWeighted += lengthRatio * score;
                timeSpokenProsWeighted += speechTime * lengthRatio * score;
            } else if (score < 0) {
                //numSpokenCons += lengthRatio;
                //timeSpokenCons += speechTime * lengthRatio;
                numSpokenConsWeighted -= lengthRatio * score;
                timeSpokenConsWeighted -= speechTime * lengthRatio * score;
            }
            if (VERBOSE) {
                System.out.println("Sentiment score: " + score);
            }
            return score;
        }

        /**
         * Reads the label for current speech (status at the end of this),
         * either from user input or from existing sources.
         *
         * @return Label
         */
        public int readLabel() {
            int l = 0;
            if (IMPORT_LABELS) {
                boolean success = true;
                try {
                    l = oldLabels.get(sessionName).get(audioId);
                } catch (NullPointerException e) {
                    // ID key doesn't exist for some reason
                    success = false;
                }
                if (success) {
                    return l;
                }
            }
            Scanner in = new Scanner(System.in);
            //System.out.print("Please enter label (0 - insufficient in both pros and cons, 1 - sufficient pros, 2 - sufficient cons, 3 - sufficient in both pros and cons): ");
            //label = in.nextInt();
            System.out.print("Please enter label: 1 for sufficient pros, 2 for sufficient cons, 3 or 12 for sufficient in both: ");
            String s = in.nextLine();
            if (s.contains("3")) {
                l = 3;
            } else {
                l = (s.contains("1") ? 1 : 0) + (s.contains("2") ? 2 : 0);
            }
            System.out.println();
            return l;
        }

        /**
         * Reads the true sentiment for current sentence,
         * either from user input or from existing sources.
         *
         * @return List of true sentiment scores (-1 or 1) for each sentence
         */
        public List<Integer> readTrueSentiment() {
            int sentenceNum = sentences.size();
            List<Integer> scores = new ArrayList<>();
            if (IMPORT_PROS_CONS) {
                boolean success = true;
                try {
                    for (int i=0; i<sentenceNum; i++) {
                        scores.add(oldSentiments.get(sessionName).get(audioId).get(i));
                    }
                } catch (NullPointerException e) {
                    // ID key doesn't exist for some reason
                    success = false;
                }
                if (success) {
                    return scores;
                }
            }
            Scanner in = new Scanner(System.in);
            String s = "";
            while (s.length() != sentenceNum) {
                System.out.print("Please enter " + sentenceNum + " sentiment scores: 1 for positive, 0 for negative, space for neutral (e.g.1 010):");
                s = in.nextLine();
            }
            for (char c : s.toCharArray()) {
                scores.add(c == '1'? 1: (c == '0'? -1: 0));
            }
            System.out.println();
            return scores;
        }

        /**
         * Write current statistics to the CSV output file.
         * Called after each speech.
         */
        public void writeCSV() throws IOException {
            List<String> l = Arrays.asList(
                    Integer.toString(sectionId),
                    Integer.toString(audioId),
                    username,
                    Double.toString(speechTime),
                    Double.toString(weightedSentiment),
                    Long.toString(numSpoken),
                    Double.toString(timeSpoken),
                    //Double.toString(numSpokenPros),
                    //Double.toString(timeSpokenPros),
                    //Double.toString(numSpokenCons),
                    //Double.toString(timeSpokenCons),
                    Double.toString(numSpokenWeighted),
                    Double.toString(timeSpokenWeighted),
                    Double.toString(numSpokenProsWeighted),
                    Double.toString(timeSpokenProsWeighted),
                    Double.toString(numSpokenConsWeighted),
                    Double.toString(timeSpokenConsWeighted),
                    Integer.toString(label)
            );
            csvWriter.append(String.join(",", l));
            csvWriter.append("\n");
        }

        /**
         * Write transcripts and sentiment scores to the CSV output file.
         * Called after each speech, but adds all sentences in separate entries.
         */
        public void writeTranscript() throws IOException {
            for (int i=0; i<sentences.size(); i++) {
                List<String> l = Arrays.asList(
                        Integer.toString(sectionId),
                        Integer.toString(audioId),
                        Integer.toString(i),
                        username,
                        "\"" + sentences.get(i) + "\"",  // Use "" to escape commas
                        Double.toString(predictedSentiments.get(i)),
                        Integer.toString(trueSentiments.get(i))
                );
                transcriptWriter.append(String.join(",", l));
                transcriptWriter.append("\n");
            }
        }
    }

    /**
     * Read labels from past CSVs, if applicable.
     * Assumes CSV file is stored in oldCSVPath and named as the session
     * name.
     * @param sessionName Name of session (room)
     */
    public static void initializeLabels(String sessionName) throws IOException {
        Path oldCSV = Paths.get(oldCSVPath).resolve(sessionName + ".csv");
        CSVReader csvReader = new CSVReader(new FileReader(oldCSV.toFile()));

        List<String> header = Arrays.asList(csvReader.readNext());
        int idHeader = header.indexOf("Speech ID");
        if (idHeader == -1) {
            idHeader = header.indexOf("ID");  // Compatibility
        }
        int labelHeader = header.indexOf("Label of current status");
        if (idHeader == -1) {
            throw new IOException("Malformatted CSV file: ID header not found");
        }
        if (labelHeader == -1) {
            throw new IOException("Malformatted CSV file: label header not found");
        }

        oldLabels.putIfAbsent(sessionName, new HashMap<>());
        String[] row = csvReader.readNext();
        while (row != null) {
            oldLabels.get(sessionName).put(Integer.parseInt(row[idHeader]),
                    Integer.parseInt(row[labelHeader]));
            row = csvReader.readNext();
        }
        csvReader.close();
    }

    /**
     * Read true sentiments from past CSVs, if applicable.
     * Assumes CSV file is stored in oldCSVPath and named as the session
     * name concatenated with "_transcript.csv".
     * @param sessionName Name of session (room)
     */
    public static void initializeTranscripts(String sessionName) throws IOException {
        Path oldCSV = Paths.get(oldCSVPath).resolve(sessionName + "_transcript.csv");
        CSVReader csvReader = new CSVReader(new FileReader(oldCSV.toFile()));

        List<String> header = Arrays.asList(csvReader.readNext());
        int speechIdHeader = header.indexOf("Speech ID");
        int sentenceIdHeader = header.indexOf("Sentence ID");
        int labelHeader = header.indexOf("True sentiment");
        if (speechIdHeader == -1) {
            throw new IOException("Malformatted CSV file: speech ID header not found");
        }
        if (sentenceIdHeader == -1) {
            throw new IOException("Malformatted CSV file: sentence ID header not found");
        }
        if (labelHeader == -1) {
            throw new IOException("Malformatted CSV file: label header not found");
        }

        oldSentiments.putIfAbsent(sessionName, new HashMap<>());
        String[] row = csvReader.readNext();
        while (row != null) {
            int speechId = Integer.parseInt(row[speechIdHeader]);
            int sentenceId = Integer.parseInt(row[sentenceIdHeader]);
            oldSentiments.get(sessionName).putIfAbsent(speechId, new HashMap<>());
            oldSentiments.get(sessionName).get(speechId).put(sentenceId,
                    Integer.parseInt(row[labelHeader]));
            row = csvReader.readNext();
        }
        csvReader.close();
    }

    /**
     * Get the time at which each section starts/ends from the JSON
     * transcript.
     * Assumes sections are contiguous, but not necessarily in order.
     * @param sectionData "sectionData" JSON array from the transcript
     * @return List of start times of each topic * that start with A *
     *      The last element is the end time of the last agenda
     */
    public static List<Long> getSectionTimes(JSONArray sectionData) {
        Set<Long> startTimes = new TreeSet<>();
        long endTime = 0;
        for (int i=0; i<sectionData.length(); i++) {
            JSONObject section = sectionData.getJSONObject(i);
            if (!section.getString("name").startsWith("A")) {
                continue;
            }
            startTimes.add(section.getLong("startTime"));
            endTime = Math.max(endTime, section.getLong("endTime"));
        }
        List<Long> times = new ArrayList<>(startTimes);
        times.add(endTime);
        return times;
    }

    /**
     * Process a single speech, i.e. an audio of one person speaking.
     * Add information of each sentence in the speech.
     * Sentences are split by the MeaningCloud API.
     *
     * @param singleSpeech Speech data as JSON object
     * @param stats Stats recorder with ongoing records from the session
     */
    public static void processSpeech(JSONObject singleSpeech, StatsRecorder stats) throws IOException {
        int id = singleSpeech.getInt("id");
        String username = singleSpeech.getString("username")
                .replaceAll(",", "");  // Prevent messing up with CSV
        long startTime = singleSpeech.getLong("startTime");
        long endTime = singleSpeech.getLong("endTime");

        // Each audio might be broken down into several sentences.
        // Combine all sentences
        JSONArray sentencesData = singleSpeech.getJSONArray("data");
        StringBuilder sentences = new StringBuilder();
        for (int j=0; j<sentencesData.length(); j++) {
            String sentence = sentencesData.getJSONObject(j).getString("text");
            // sentencesData.getJSONObject(j).getDouble("confidence");  // For future use
            sentences.append(sentence);
        }

        String speech = sentences.toString();
        if (speech.length() == 0 && IGNORE_EMPTY_MESSAGES) {
            return;
        }

        stats.addSpeech(id, username, speech, startTime, endTime);
    }

    /**
     * Process a session (room) and write all generated data points to a CSV file.
     * @param sessionName
     * @param sessionData
     */
    public static void processSession(String sessionName, JSONObject sessionData) throws IOException {
        JSONArray sessionAudio = sessionData.getJSONArray("audioData");
        List<Long> sectionTimes = getSectionTimes(sessionData.getJSONArray("sectionData"));
        if (sectionTimes.isEmpty()) {
            return;
        }

        System.out.println(sessionName);
        System.out.println("---------");

        if (IMPORT_LABELS) {  // IMPORTANT: Do this before writing the new CSV file since it might override old data
            initializeLabels(sessionName);
        }
        if (IMPORT_PROS_CONS) {  // IMPORTANT: Do this before writing the new CSV file since it might override old data
            initializeTranscripts(sessionName);
        }

        File file = new File(sessionName + ".csv");
        file.createNewFile();
        FileWriter csvWriter = new FileWriter(file);
        File transcriptFile = new File(sessionName + "_transcript.csv");
        transcriptFile.createNewFile();
        FileWriter transcriptWriter = new FileWriter(transcriptFile);

        // Create column labels for csv
        csvWriter.append(String.join(",", headers) + "\n");
        transcriptWriter.append(String.join(",", transcriptHeaders) + "\n");

        int currentSection = 0;
        StatsRecorder stats = new StatsRecorder(sessionName, csvWriter, transcriptWriter);

        for (int i=0; i<sessionAudio.length(); i++){
            JSONObject singleSpeech = sessionAudio.getJSONObject(i);
            long startTime = singleSpeech.getLong("startTime");

            // Checks if a new section needs to be started, and whether
            // this speech should count at all
            if (startTime < sectionTimes.get(currentSection)) {
                continue;  // Intro (before A1)
            } else if (startTime >= sectionTimes.get(currentSection + 1)) {
                currentSection += 1;
                if (currentSection == sectionTimes.size() - 1) {
                    break;  // Past the last section (question generation, sorting)
                }
                stats.newSection();
                System.out.println("---------");
                System.out.println("New section");
                System.out.println("---------");
            }

            processSpeech(singleSpeech, stats);
        }
        System.out.println();
        System.out.println();
        System.out.println();
        csvWriter.close();
        transcriptWriter.close();
    }

    public static void main(String[] args) throws IOException {
        System.out.print("Do you want to input conversation labels manually? (Y/N):");
        Scanner in = new Scanner(System.in);
        String s = in.nextLine();
        IMPORT_LABELS = !(s.equals("Y"));
        System.out.print("Do you want to input true sentiment (pros/cons) labels manually? (Y/N):");
        s = in.nextLine();
        IMPORT_PROS_CONS = !(s.equals("Y"));

        JSONObject database = null;
        InputStream is = jsonParser.class.getResourceAsStream(path);
        if (is == null) {
            throw new FileNotFoundException("Cannot find resource file " + path);
        }
        JSONTokener tokener = new JSONTokener(is);
        database = new JSONObject(tokener);

		for (Object sessionObj : database.keySet()) {
            String sessionName = (String) sessionObj;
            JSONObject sessionData = database.getJSONObject(sessionName);
            processSession(sessionName, sessionData);
        }

        System.out.println("Remaining API credits: " + credits);
    } 
} 
