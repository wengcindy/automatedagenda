import java.io.FileReader; 
import java.util.Iterator; 
import java.util.Map; 
  
import org.json.simple.JSONArray; 
import org.json.simple.JSONObject; 
import org.json.simple.parser.*; 
  
public class jsonParser 
{ 
    public static void main(String[] args) 
    { 
        // parsing file "JSONExample.json" 
        Object obj = new JSONParser().parse(new FileReader("2019winter.json")); 
          
        // typecasting obj to JSONObject 
        JSONObject jo = (JSONObject) obj; 

        JSONObject w2 = (JSONObject) jo.getJSONObject("2019Winter2");
        JSONArray w2_audio = (JSONArray) w2.getJSONArray("audioData");
        for(int i = 0; i < w2_audio.length(); i++){
            String username = (String) w2_audio[i].get("username");
            System.out.println(username);
        }



          
        // // getting firstName and lastName 
        // String firstName = (String) jo.get("firstName"); 
        // String lastName = (String) jo.get("lastName"); 
          
        // System.out.println(firstName); 
        // System.out.println(lastName); 
          
        // // getting age 
        // long age = (long) jo.get("age"); 
        // System.out.println(age); 
          
        // // getting address 
        // Map address = ((Map)jo.get("address")); 
          
        // // iterating address Map 
        // Iterator<Map.Entry> itr1 = address.entrySet().iterator(); 
        // while (itr1.hasNext()) { 
        //     Map.Entry pair = itr1.next(); 
        //     System.out.println(pair.getKey() + " : " + pair.getValue()); 
        // } 
          
        // // getting phoneNumbers 
        // JSONArray ja = (JSONArray) jo.get("phoneNumbers"); 
          
        // // iterating phoneNumbers 
        // Iterator itr2 = ja.iterator(); 
          
        // while (itr2.hasNext())  
        // { 
        //     itr1 = ((Map) itr2.next()).entrySet().iterator(); 
        //     while (itr1.hasNext()) { 
        //         Map.Entry pair = itr1.next(); 
        //         System.out.println(pair.getKey() + " : " + pair.getValue()); 
        //     } 
        // } 
    } 
} 