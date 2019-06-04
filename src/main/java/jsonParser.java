import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.json.*;
import org.json.simple.JSONArray; 
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import org.json.simple.parser.*; 
  
public class jsonParser 
{ 
    public static void main(String[] args) throws IOException
    { 

        Object obj = null;
        JSONObject database = null;
        try {
			obj = new JSONParser().parse(new FileReader("2019winter.json"));
			database = (JSONObject) obj;
		} catch (IOException | ParseException | ClassCastException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

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
            JSONObject sessionData = (JSONObject) database.get(sessionName);
            JSONArray sessionAudio = (JSONArray) sessionData.get("audioData");
            for(Object o: sessionAudio){
                JSONObject singleSpeech = (JSONObject) o;
                String username = (String) singleSpeech.get("username");
                double startTime = ((long) singleSpeech.get("startTime"))/1000.0;
                double endTime = ((long) singleSpeech.get("endTime"))/1000.0;
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
    } 
} 
