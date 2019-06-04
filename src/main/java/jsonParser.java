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

            JSONObject result = response.getBody().getObject();
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
            csvWriter.append("Name");
            csvWriter.append(",");
            csvWriter.append("Individual speaking time");
            csvWriter.append(",");
            csvWriter.append("Total number of people spoken");
            csvWriter.append(",");
            csvWriter.append("Total speaking time");
            csvWriter.append("\n");

            int numspoken = 0;
            double timespoken = 0;
            JSONObject sessionData = database.getJSONObject(sessionName);
            JSONArray sessionAudio = sessionData.getJSONArray("audioData");

            for (int i=0; i<sessionAudio.length(); i++){
                JSONObject singleSpeech = sessionAudio.getJSONObject(i);
                String username = singleSpeech.getString("username");
                double startTime = singleSpeech.getLong("startTime")/1000.0;
                double endTime = singleSpeech.getLong("endTime")/1000.0;
                numspoken++;
                timespoken += endTime - startTime;

                System.out.println(username);
                System.out.println(endTime - startTime);
                System.out.println("Num of ppl spoken: " + numspoken);
                System.out.println("Total time spoken: " + timespoken);
                System.out.println();

                //Write to csv file
                List<String> l = Arrays.asList(username, Double.toString(endTime-startTime), Long.toString(numspoken), Double.toString(timespoken));
                csvWriter.append(String.join(",", l));
                csvWriter.append("\n");
            }
            System.out.println();
            System.out.println();
            System.out.println();
            csvWriter.close();
        }

        /*String str = "Unirest is a set of lightweight HTTP libraries available in multiple languages, built and maintained by Mashape, who also maintain the open-source API Gateway Kong. ";
        SentimentAnalyzer s = new SentimentAnalyzer(str);
        System.out.println(s.sentimentTag);
        System.out.println(s.sentimentScore);
        System.out.println(s.sentimentConfidence);*/
    } 
} 
