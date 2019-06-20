import org.apache.commons.io.IOUtils;
import org.python.core.PyException;
import org.python.core.PyInteger;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;
import org.json.*;

import java.util.Properties;

public class PythonRunnerTrial {
    private static PythonInterpreter intr;

    public static void main(String[] args) {

        Properties props = new Properties();
        try {
            Process p = Runtime.getRuntime().exec(new String[]{
                    "python", "-c", "import json; import sys; print(json.dumps(sys.path))"});
            p.waitFor();

            String stdout = IOUtils.toString(p.getInputStream());
            JSONArray syspathRaw = new JSONArray(stdout);
            for (int i = 0; i < syspathRaw.length(); i++) {
                String path = syspathRaw.getString(i);
                System.out.println(path);
                if (true) { //(path.contains("site-packages") || path.contains("dist-packages")) {
                    //intr.exec(String.format("sys.path.append('%s')", path));
                    //props.put("python.home", path.replaceAll("site-packages", "").replaceAll("dist-packages", ""));
                    //props.put("python.home", "C:\\Users\\shift\\AppData\\Local\\Programs\\Python\\Python37-32");
                    //System.out.println(path);
                }
            }
        } catch (Exception ex) {ex.printStackTrace();}
        Properties preprops = System.getProperties();
        PythonInterpreter.initialize(preprops, props, new String[0]);
        intr =  new PythonInterpreter();

        //intr.exec("import sys");
        //intr.exec("print(sys.path)");
        //intr.exec("import gensim");
    }
}
