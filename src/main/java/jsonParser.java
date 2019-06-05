import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;*/
import kong.unirest.HttpResponse;
import kong.unirest.JsonNode;
import kong.unirest.Unirest;
import kong.unirest.UnirestException;
import org.json.*;
/*import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import org.json.simple.parser.*;*/
  
public class jsonParser {
    static String path = "2019winter.json";
    static int credits = 0;

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

            System.out.println(result);
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
            System.out.println(sessionName);
            System.out.println("---------");

            File file = new File(sessionName + ".csv");
            file.createNewFile();
            FileWriter csvWriter = new FileWriter(file);

            // Create column labels for csv
            List<String> headers = Arrays.asList(
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
            csvWriter.append(String.join(",", headers) + "\n");

            int numSpoken = 0;
            double timeSpoken = 0;
            double numSpokenPros = 0, timeSpokenPros = 0;
            double numSpokenCons = 0, timeSpokenCons = 0;
            double numSpokenWeighted = 0, timeSpokenWeighted = 0;
            double numSpokenProsWeighted = 0, timeSpokenProsWeighted = 0;
            double numSpokenConsWeighted = 0, timeSpokenConsWeighted = 0;
            JSONObject sessionData = database.getJSONObject(sessionName);
            JSONArray sessionAudio = sessionData.getJSONArray("audioData");

            for (int i=0; i<sessionAudio.length(); i++){
                JSONObject singleSpeech = sessionAudio.getJSONObject(i);
                String username = singleSpeech.getString("username");
                double startTime = singleSpeech.getLong("startTime")/1000.0;
                double endTime = singleSpeech.getLong("endTime")/1000.0;
                double speechTime = endTime - startTime;
                double weightedSentiment = 0;
                numSpoken++;
                timeSpoken += speechTime;

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

                /* Analyze the sentiment of each sentence, and then increment
                 the # people, speaking time and weighted sentiment score
                 accordingly.
                 The increase is according to the proportion this sentence
                 takes in the entire speech in terms of characters.
                 For example, a 40-character sentence about pros in the
                 entire speech of 100 characters will contribute 0.4 to
                 the number of people spoken about pros.*/
                for (String sentence : sentences) {
                    SentimentAnalyzer sa = new SentimentAnalyzer(sentence);
                    int score = sa.sentimentScore;
                    double ratio = sentence.length() * 1.0 / totalLength;
                    weightedSentiment += ratio * score;
                    numSpokenWeighted += ratio * score;
                    timeSpokenWeighted += speechTime * ratio * score;
                    if (score > 0) {
                        numSpokenPros += ratio;
                        timeSpokenPros += speechTime * ratio;
                        numSpokenProsWeighted += ratio * score;
                        timeSpokenProsWeighted += speechTime * ratio * score;
                    } else if (score < 0) {
                        numSpokenCons += ratio;
                        timeSpokenCons += speechTime * ratio;
                        numSpokenConsWeighted -= ratio * score;
                        timeSpokenConsWeighted -= speechTime * ratio * score;
                    }
                }

                System.out.println(username);
                System.out.println("Time of this speech:" + speechTime);
                System.out.println("Weighted sentiment score:" + weightedSentiment);
                System.out.println("Num of ppl spoken: " + numSpoken);
                System.out.println("Total time spoken: " + timeSpoken);
                System.out.println("Num of ppl spoken weighted: " + numSpokenWeighted);
                System.out.println("Total time spoken weighted: " + timeSpokenWeighted);
                System.out.println("Num of ppl spoken pros: " + numSpokenPros);
                System.out.println("Total time spoken pros: " + timeSpokenPros);
                System.out.println("Num of ppl spoken cons: " + numSpokenCons);
                System.out.println("Total time spoken cons: " + timeSpokenCons);
                System.out.println("Num of ppl spoken pros weighted: " + numSpokenProsWeighted);
                System.out.println("Total time spoken pros weighted: " + timeSpokenProsWeighted);
                System.out.println("Num of ppl spoken cons weighted: " + numSpokenConsWeighted);
                System.out.println("Total time spoken cons weighted: " + timeSpokenConsWeighted);
                System.out.println();

                // Write to csv file
                List<String> l = Arrays.asList(
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
            System.out.println();
            System.out.println();
            System.out.println();
            csvWriter.close();
        }

        System.out.println("Remaining API credits: " + credits);
    } 
} 
