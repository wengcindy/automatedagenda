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
		try {
			obj = new JSONParser().parse(new FileReader("2019winter.json"));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		
		// Set up file and file writer
		File winter1 = new File("winter1.csv");
		winter1.createNewFile();
		File winter2 = new File("winter2.csv");
		winter2.createNewFile();
		File winter3 = new File("winter3.csv");
		winter3.createNewFile();
        FileWriter csvWriter1 = new FileWriter(winter1); 
        FileWriter csvWriter2 = new FileWriter(winter2); 
        FileWriter csvWriter3 = new FileWriter(winter3); 

        
        // Create column labels for csv
        csvWriter1.append("Name");  
        csvWriter1.append(",");  
        csvWriter1.append("Individual speaking time");  
        csvWriter1.append(",");  
        csvWriter1.append("Total number of people spoken");  
        csvWriter1.append(",");  
        csvWriter1.append("Total speaking time");   
        csvWriter1.append("\n");
        
        csvWriter2.append("Name");  
        csvWriter2.append(",");  
        csvWriter2.append("Individual speaking time");  
        csvWriter2.append(",");  
        csvWriter2.append("Total number of people spoken");  
        csvWriter2.append(",");  
        csvWriter2.append("Total speaking time");   
        csvWriter2.append("\n");
        
        csvWriter3.append("Name");  
        csvWriter3.append(",");  
        csvWriter3.append("Individual speaking time");  
        csvWriter3.append(",");  
        csvWriter3.append("Total number of people spoken");  
        csvWriter3.append(",");  
        csvWriter3.append("Total speaking time");   
        csvWriter3.append("\n");

        JSONObject jo = (JSONObject) obj; 
        
        List l = new ArrayList();
        int numspoken = 0;
        int timespoken = 0;
        
        // Winter 1
        System.out.println("Winter 1");
        System.out.println("---------");
        JSONObject w1 = (JSONObject) jo.get("2019winter1");
        JSONArray w1_audio = (JSONArray) w1.get("audioData");
        for(Object o: w1_audio){
            String username = (String) ((JSONObject) o).get("username");
            long startTime = ((long) ((JSONObject)o).get("startTime"))/1000;
            long endTime = ((long) ((JSONObject)o).get("endTime"))/1000;
            numspoken++;
            timespoken += endTime - startTime;
            
            System.out.println(username);
            System.out.println(endTime - startTime);
            System.out.println("Num of ppl spoken: " + numspoken);
            System.out.println("Total time spoken: " + timespoken);
            System.out.println();
            
            //Write to csv file
            l = Arrays.asList(username, Long.toString(endTime-startTime), Long.toString(numspoken), Long.toString(timespoken));
            csvWriter1.append(String.join(",", l));
            csvWriter1.append("\n");
        }
        System.out.println();
        System.out.println();
        System.out.println();
        
        numspoken = 0;
        timespoken = 0;
        
        // Winter 2
        System.out.println("Winter 2");
        System.out.println("---------");
        JSONObject w2 = (JSONObject) jo.get("2019winter2");
        JSONArray w2_audio = (JSONArray) w2.get("audioData");
        for(Object o: w2_audio){
            String username = (String) ((JSONObject) o).get("username");
            long startTime = ((long) ((JSONObject)o).get("startTime"))/1000;
            long endTime = ((long) ((JSONObject)o).get("endTime"))/1000;
            numspoken++;
            timespoken += endTime - startTime;
            System.out.println(username);
            System.out.println(endTime - startTime);
            System.out.println("Num of ppl spoken: " + numspoken);
            System.out.println("Total time spoken: " + timespoken);
            System.out.println();
            
            l = Arrays.asList(username, Long.toString(endTime-startTime), Long.toString(numspoken), Long.toString(timespoken));
            csvWriter2.append(String.join(",", l));
            csvWriter2.append("\n");
        }
        System.out.println();
        System.out.println();
        System.out.println();
        
        numspoken = 0;
        timespoken = 0;
        
        // Winter 3
        System.out.println("Winter 3");
        System.out.println("---------");
        JSONObject w3 = (JSONObject) jo.get("2019winter3");
        JSONArray w3_audio = (JSONArray) w3.get("audioData");
        for(Object o: w3_audio){
            String username = (String) ((JSONObject) o).get("username");
            long startTime = ((long) ((JSONObject)o).get("startTime"))/1000;
            long endTime = ((long) ((JSONObject)o).get("endTime"))/1000;
            numspoken++;
            timespoken += endTime - startTime;
            System.out.println(username);
            System.out.println(endTime - startTime);
            System.out.println("Num of ppl spoken: " + numspoken);
            System.out.println("Total time spoken: " + timespoken);
            System.out.println();
            
            l = Arrays.asList(username, Long.toString(endTime-startTime), Long.toString(numspoken), Long.toString(timespoken));
            csvWriter3.append(String.join(",", l));
            csvWriter3.append("\n");
        }
       
        csvWriter1.close();
        csvWriter2.close();
        csvWriter3.close();
    } 
} 
