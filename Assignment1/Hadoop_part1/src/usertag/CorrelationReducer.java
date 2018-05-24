package usertag;
import java.io.IOException;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
/**
 * inout: (category->countrycode+videoid) output:(category: total: xxx; xx% in xxx)
 * @author Yuanqi Pang
 *
 */
public class CorrelationReducer extends Reducer<Text, Text, Text, Text> {
	Text result = new Text();

	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
		// get parameter
		Configuration conf = context.getConfiguration();
		String countrya = conf.get("countrya");
		String countryb = conf.get("countryb");
		HashMap<String, Integer> countryavideo = new HashMap<String, Integer>();
		HashMap<String, Integer> countrybvideo = new HashMap<String, Integer>();
		for(Text text: values) {
			String cv = text.toString();
			String countrycode = cv.substring(0,2);
			if(countrycode.equals(countrya)) {
			    if(!countryavideo.containsKey(cv)){
				countryavideo.put(cv,1);
                            }
			}
			else {
				countrybvideo.put(cv,1);
			}
		}
        int sumA = countryavideo.size(),sumB = 0;
	for(String a:countryavideo.keySet()) {
	    for(String b:countrybvideo.keySet()) {
			if(a.substring(2).equals(b.substring(2))) {
		    	sumB++;
                break;
			}
	    }
	}
	double percentage = 0.0;
	if(sumB != 0) {
	    percentage = (double)sumB/(double)sumA;
	}
	String per = String.format("%.1f",percentage*100);
	result.set("; total: "+String.valueOf(sumA)+" ; "+per+"% in "+countryb);
	context.write(key,result);
	}
}
