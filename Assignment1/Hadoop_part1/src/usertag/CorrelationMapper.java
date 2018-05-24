package usertag;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
/**
 * input: csv string, output(category->countrycode+videoid)
 * @author yuanqi pang
 *
 */
public class CorrelationMapper extends Mapper<Object, Text, Text, Text> {
	private Text category = new Text(), info = new Text();
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		String countrya = conf.get("countrya");
		String countryb = conf.get("countryb");
		String[] dataArray = value.toString().split(",");
        int a = dataArray.length;
        String ct = dataArray[a-1].trim();
        String vd = dataArray[0].trim();
		if (a == 18) {
			if (ct.equals(countrya) | ct.equals(countryb)) {
				category.set(dataArray[5].trim());
				info.set(ct+vd);
				context.write(category, info);
			}
		} else {
			String[] shabi = value.toString().split("000Z");
			String[] shabi1 = shabi[0].split(",");
			if (ct.equals(countrya) | ct.equals(countryb)) {
				category.set(shabi1[shabi1.length - 2].trim());
				info.set(ct+vd);
				context.write(category, info);
			}
		}
		// String[] dataArray = value.toString().split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");
		// int a = dataArray.length;
		// String ct = dataArray[a-1].trim();
  //       String vd = dataArray[0].trim();
  //       if (ct.equals(countrya) | ct.equals(countryb)) {
  //       	category.set(dataArray[5].trim());
  //       	info.set(ct+vd);
  //       	context.write(category,info);
  //       }
	}
}
