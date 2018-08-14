package com.spark.clustering;

import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LDAClustering {

    public static void getTopics(SparkSession spark) {

        Dataset<Row> dataset = spark.read().format("libsvm").load("sample_lda_libsvm_data.txt");

        LDA lda = new LDA().setK(10).setMaxIter(10);
        LDAModel model = lda.fit(dataset);

        model.logLikelihood(dataset);
        model.logPerplexity(dataset);

        Dataset<Row> topics = model.describeTopics(3);

        topics.show(false);

        Dataset<Row> transformed = model.transform(dataset);
        transformed.show(false);
    }
}
