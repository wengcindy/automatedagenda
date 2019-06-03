import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator; 
import java.util.Map; 

import org.json.*;
import org.json.simple.JSONArray; 
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import org.json.simple.parser.*; 
  
public class jsonParser 
{ 
    public static void main(String[] args) 
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
           
        JSONObject jo = (JSONObject) obj; 
        
        System.out.println("Winter 1");
        System.out.println("---------");
        JSONObject w1 = (JSONObject) jo.get("2019winter1");
        JSONArray w1_audio = (JSONArray) w1.get("audioData");
        for(Object o: w1_audio){
            String username = (String) ((JSONObject) o).get("username");
            long startTime = (long) ((JSONObject)o).get("startTime");
            long endTime = (long) ((JSONObject)o).get("endTime");
            System.out.println(username);
            System.out.println(endTime - startTime);
            System.out.println();
        }
        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println("Winter 2");
        System.out.println("---------");
        JSONObject w2 = (JSONObject) jo.get("2019winter2");
        JSONArray w2_audio = (JSONArray) w2.get("audioData");
        for(Object o: w2_audio){
            String username = (String) ((JSONObject) o).get("username");
            long startTime = (long) ((JSONObject)o).get("startTime");
            long endTime = (long) ((JSONObject)o).get("endTime");
            System.out.println(username);
            System.out.println(endTime - startTime);
            System.out.println();
        }
        System.out.println();
        System.out.println();
        System.out.println();
        
        System.out.println("Winter 3");
        System.out.println("---------");
        JSONObject w3 = (JSONObject) jo.get("2019winter3");
        JSONArray w3_audio = (JSONArray) w3.get("audioData");
        for(Object o: w2_audio){
            String username = (String) ((JSONObject) o).get("username");
            long startTime = (long) ((JSONObject)o).get("startTime");
            long endTime = (long) ((JSONObject)o).get("endTime");
            System.out.println(username);
            System.out.println(endTime - startTime);
            System.out.println();
        }
    } 
} 
