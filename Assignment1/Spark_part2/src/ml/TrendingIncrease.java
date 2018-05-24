package ml;
import java.util.List;
import java.util.ArrayList;
import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;
import com.google.common.collect.Lists;
import org.apache.commons.collections.IteratorUtils;
import scala.Tuple2;
import org.apache.spark.SparkContext;
 /*
 * <!-- spark-submit --class ml.MovieLensLarge --master yarn-cluster sparkML.jar
 * /share/movie/small/ week5_out/ -->
 * @author Yuanqi Pang
 */
public class TrendingIncrease {
	public static void main(String[] args) {
		String inputDataPath = args[0], outputDataPath = args[1];
		SparkConf conf = new SparkConf();
		conf.setAppName("Video trending app");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<String> data = sc.textFile(inputDataPath);
		JavaPairRDD<String, Integer> vvExtraction = data.mapToPair(s -> {
			String[] values = s.split(",");
			String[] a = s.split("000Z");
			if (a.length == 2) {
				return new Tuple2<String, Integer>(values[values.length - 1]+ values[0],
						Integer.parseInt(a[1].split(",")[2]));
			} else {
				return new Tuple2<String, Integer>("row1", 0);
			}
		});
		JavaPairRDD<String, Double> filtered = vvExtraction.groupByKey().filter(v->itersize(v._2)>=2).mapValues(r->percent(r)).filter(v->v._2>=1000.0);
		JavaPairRDD<String, String> sorted = filtered.mapToPair(r->r.swap()).sortByKey(false).mapToPair(r->r.swap()).mapValues(v->String.format("%.1f",v)+"%");
		JavaPairRDD<String, String> output = sorted.mapToPair(s->{return new Tuple2<String,String>(s._1.substring(0,2),s._1.substring(2)+", "+s._2);}).sortByKey(false);
		output.saveAsTextFile(outputDataPath);
		sc.close();
	}
	public static int itersize(Iterable<Integer> a) {
		int ct = 0;
		for(Integer x:a) {
			ct++;
		}
		return ct;
	}
	public static double percent(Iterable<Integer> a) {
		List<Integer> views = Lists.newArrayList(a);
		double view1 = (double)views.get(0);
		double view2 = (double)views.get(1);
		double percent = ((view2-view1)*100)/view1;
		return percent;
	}
}
