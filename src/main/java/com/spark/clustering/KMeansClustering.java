package com.spark.clustering;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KMeansClustering {


    public KMeansClustering() {
    }

    public static void getClusterCenters(SparkSession spark) {


        Dataset<Row> dataset = spark.read().format("libsvm").load("sample_kmeans_data.txt");

        KMeans kMeans = new KMeans().setK(2).setSeed(1L);
        KMeansModel model = kMeans.fit(dataset);

        double WSSSE = model.computeCost(dataset);
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

        Vector[] centers = model.clusterCenters();

        System.out.println("Cluster Centers: ");

        for(Vector vector: centers) {
            System.out.println(vector);
        }

    }


}
