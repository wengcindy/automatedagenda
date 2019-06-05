import java.io.*;
import java.util.*;

import kong.unirest.HttpResponse;
import kong.unirest.JsonNode;
import kong.unirest.Unirest;
import kong.unirest.UnirestException;
import org.json.*;
  
public class jsonParser {
    final static boolean VERBOSE = true;
    final static boolean IGNORE_EMPTY_MESSAGES = true;
    static String path = "2019winter.json";

    static int credits = 0;

    static List<String> headers = Arrays.asList(
            "ID",
            "Name",
            "Individual speaking time",
            "Individual weighted sentiment score",
            "Total number of people spoken",
            "Total speaking time",
            "Total number of people spoken about pros",
            "Total speaking time about pros",
            "Total number of people spoken about cons",
            "Total speaking time about cons",
            "Weighted total number of people spoken",
            "Weighted total speaking time",
            "Weighted total number of people spoken about pros",
            "Weighted total speaking time about pros",
            "Weighted total number of people spoken about cons",
            "Weighted total speaking time about cons"
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
        boolean isSubjective;  // Subjective/objective tags from sentiment analysis
        int sentimentConfidence;  // int 0-100

        public SentimentAnalyzer(String msg) {
            initMessage(msg);
        }

        public void initMessage(String msg) {
            message = msg;
            // Check if the message is empty (has no letters, probably due to mic failure)
            if (msg.replaceAll("[^a-zA-Z0-9]","").equals("")) { // https://stackoverflow.com/questions/4945695/how-to-filter-string-for-unwanted-characters-using-regex
                isEmpty = true;
                return;
            }

            sentimentAnalysis();
        }

        private void sentimentAnalysis() {
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
                    return;
                }
                if (!response.isSuccess()) {
                    System.out.println("Error " + response.getStatus() + " when running Sentiment Analysis: " + response.getStatusText());
                    return;
                }
                result = response.getBody().getObject();
                String msg = result.getJSONObject("status").getString("msg");
                if (!msg.equals("OK")) {
                    if (msg.equals("Request rate limit exceeded")) {
                        result = null;
                    } else {
                        System.out.println("API error when running Sentiment Analysis: " + msg);
                        return;
                    }
                } else {
                    credits = result.getJSONObject("status").getInt("remaining_credits");
                }
            }

            //System.out.println(result);
            sentimentTag = result.getString("score_tag");
            // Convert letter score to numeric score
            List<String> sentiments = Arrays.asList("N+", "N", "NEU", "P", "P+");
            sentimentScore = sentiments.indexOf(sentimentTag) - 2;
            if (sentimentScore == -3) sentimentScore = 0;  // NONE
            isAgreement = result.getString("agreement").equals("AGREEMENT");
            isSubjective = result.getString("subjectivity").equals("SUBJECTIVE");
            sentimentConfidence = result.getInt("confidence");
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
        // For current speech
        String username = null;
        int audioId = 0;
        double speechTime = 0;
        double weightedSentiment = 0;
        // For the entire section
        int numSpoken = 0;
        double timeSpoken = 0;
        double numSpokenPros = 0, timeSpokenPros = 0;
        double numSpokenCons = 0, timeSpokenCons = 0;
        double numSpokenWeighted = 0, timeSpokenWeighted = 0;
        double numSpokenProsWeighted = 0, timeSpokenProsWeighted = 0;
        double numSpokenConsWeighted = 0, timeSpokenConsWeighted = 0;
        FileWriter csvWriter = null;

        public StatsRecorder(FileWriter csvWriter) {
            this.csvWriter = csvWriter;
            newSection();
        }

        public void newSection() {
            String username = null;
            audioId = 0;
            speechTime = 0;
            weightedSentiment = 0;
            numSpoken = 0;
            timeSpoken = 0;
            numSpokenPros = 0; timeSpokenPros = 0;
            numSpokenCons = 0; timeSpokenCons = 0;
            numSpokenWeighted = 0; timeSpokenWeighted = 0;
            numSpokenProsWeighted = 0; timeSpokenProsWeighted = 0;
            numSpokenConsWeighted = 0; timeSpokenConsWeighted = 0;
        }

        /**
         * Start a new speech, i.e. an audio of one person speaking.
         * @param id ID of speech from JSON (for importing labels)
         * @param user Name of current speaker
         * @param startTime Start time of this speech, as long nubmer
         * @param endTime End time of this speech, as long nubmer
         */
        public void newSpeech(int id, String user, long startTime, long endTime) {
            username = user;
            audioId = id;
            speechTime = ((double) endTime - (double) startTime) / 1000;
            weightedSentiment = 0;
            numSpoken++;
            timeSpoken += speechTime;
        }

        /**
         * Add a new sentence. Runs sentiment analysis and increment
         * corresponding counters.
         *
         * The increase is according to the proportion this sentence
         * takes in the person's entire speech in terms of characters.
         * For example, a 40-character sentence about pros in the
         * entire speech of 100 characters will contribute 0.4 to
         * the number of people spoken about pros.
         * @param sentence Sentence text
         * @param lengthRatio Proportion of this sentence's length to the
         *                    total length of that person's speech
         * @return Score of this sentence
         */
        public int addSentence(String sentence, double lengthRatio) {
            SentimentAnalyzer sa = new SentimentAnalyzer(sentence);
            int score = sa.sentimentScore;
            weightedSentiment += lengthRatio * score;
            numSpokenWeighted += lengthRatio * score;
            timeSpokenWeighted += speechTime * lengthRatio * score;
            if (score > 0) {
                numSpokenPros += lengthRatio;
                timeSpokenPros += speechTime * lengthRatio;
                numSpokenProsWeighted += lengthRatio * score;
                timeSpokenProsWeighted += speechTime * lengthRatio * score;
            } else if (score < 0) {
                numSpokenCons += lengthRatio;
                timeSpokenCons += speechTime * lengthRatio;
                numSpokenConsWeighted -= lengthRatio * score;
                timeSpokenConsWeighted -= speechTime * lengthRatio * score;
            }
            if (VERBOSE) {
                System.out.println("Sentiment score: " + score);
            }
            return score;
        }

        /**
         * Write current statistics to the CSV output file.
         * Called after each speech.
         */
        public void writeCSV() throws IOException {
            List<String> l = Arrays.asList(
                    Integer.toString(audioId),
                    username,
                    Double.toString(speechTime),
                    Double.toString(weightedSentiment),
                    Long.toString(numSpoken),
                    Double.toString(timeSpoken),
                    Double.toString(numSpokenPros),
                    Double.toString(timeSpokenPros),
                    Double.toString(numSpokenCons),
                    Double.toString(timeSpokenCons),
                    Double.toString(numSpokenWeighted),
                    Double.toString(timeSpokenWeighted),
                    Double.toString(numSpokenProsWeighted),
                    Double.toString(timeSpokenProsWeighted),
                    Double.toString(numSpokenConsWeighted),
                    Double.toString(timeSpokenConsWeighted)
            );
            csvWriter.append(String.join(",", l));
            csvWriter.append("\n");
        }
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
     * @param singleSpeech Speech data as JSON object
     * @param stats Stats recorder with ongoing records from the session
     */
    public static void processSpeech(JSONObject singleSpeech, StatsRecorder stats) throws IOException {
        int id = singleSpeech.getInt("id");
        String username = singleSpeech.getString("username");
        long startTime = singleSpeech.getLong("startTime");
        long endTime = singleSpeech.getLong("endTime");

        // Each audio might be broken down into several sentences.
        // Extract all sentences
        JSONArray sentencesData = singleSpeech.getJSONArray("data");
        List<String> sentences = new ArrayList<>();
        int totalLength = 0;
        for (int j=0; j<sentencesData.length(); j++) {
            String sentence = sentencesData.getJSONObject(j).getString("text");
            // sentencesData.getJSONObject(j).getDouble("confidence");  // For future use
            sentences.add(sentence);
            totalLength += sentence.length();
        }

        if (totalLength == 0 && IGNORE_EMPTY_MESSAGES) {
            return;
        }

        stats.newSpeech(id, username, startTime, endTime);
        for (String sentence : sentences) {
            double ratio = sentence.length() * 1.0 / totalLength;
            System.out.println(sentence);
            int score = stats.addSentence(sentence, ratio);
            System.out.println();

            // Labeling tools to be added
        }
        stats.writeCSV();
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
        File file = new File(sessionName + ".csv");
        file.createNewFile();
        FileWriter csvWriter = new FileWriter(file);

        // Create column labels for csv
        csvWriter.append(String.join(",", headers) + "\n");

        int currentSection = 0;
        StatsRecorder stats = new StatsRecorder(csvWriter);

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
    }

    public static void main(String[] args) throws IOException {
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
