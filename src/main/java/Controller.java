import com.spark.clustering.KMeansClustering;
import com.spark.clustering.LDAClustering;
import org.apache.spark.sql.SparkSession;

public class Controller {

    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder().appName("Spark-ML").master("local[*]").getOrCreate();

        //KMeansClustering.getClusterCenters(spark);
        LDAClustering.getTopics(spark);

    }
}
