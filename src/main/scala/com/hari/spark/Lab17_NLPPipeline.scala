package com.hari.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}
// import com.hari.spark.Utils._

object Lab17_NLPPipeline {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // 1. --- Read Dataset ---
    val dataPath = "D:/ScalaProjects/nlp/spark_labs/data/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = spark.read.json(dataPath).limit(1000) // Limit for faster processing during lab
    println(s"Successfully read ${initialDF.count()} records.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = false) // Show full content for better understanding

    // --- Pipeline Stages Definition ---

    // 2. --- Tokenization ---
    
	/*
	val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']") // Fix: Use \\s for regex, and \" for double quote
	*/


    // Alternative Tokenizer: A simpler, whitespace-based tokenizer.
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")

	val tokenized = tokenizer.transform(initialDF)
	tokenized.select("text", "tokens").show(5, false)


    // 3. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

	val removed = stopWordsRemover.transform(tokenized)
	removed.select("tokens", "filtered_tokens").show(5, false)


    // 4. --- Vectorization (Term Frequency) ---
    // Convert tokens to feature vectors using HashingTF (a fast way to do count vectorization).
    // setNumFeatures defines the size of the feature vector. This is the maximum number of features
    // (dimensions) in the output vector. Each word is hashed to an index within this range.
    //
    // If setNumFeatures is smaller than the actual vocabulary size (number of unique words),
    // hash collisions will occur. This means different words will map to the same feature index.
    // While this leads to some loss of information, it allows for a fixed, manageable vector size
    // regardless of how large the vocabulary grows, saving memory and computation for very large datasets.
    // 20,000 is a common starting point for many NLP tasks.
    
/*
	val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(1000) // Set the size of the feature vector

    // 5. --- Vectorization (Inverse Document Frequency) ---
    
	val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features")
	
	val featurizedData = hashingTF.transform(removed)

	// TF result (raw_features)
	featurizedData.select("filtered_tokens", "raw_features").show(5, false)

	val idfModel = idf.fit(featurizedData)
	val rescaledData = idfModel.transform(featurizedData)

	// TF-IDF result (features)
	rescaledData.select("filtered_tokens", "features").show(5, false)

*/


	import org.apache.spark.ml.feature.Word2Vec

	val word2Vec = new Word2Vec()
	  .setInputCol("filtered_tokens")
	  .setOutputCol("features")
	  .setVectorSize(100)
	  .setMinCount(0)

//	val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, word2Vec))


//Normalizer

	import org.apache.spark.ml.feature.Normalizer

	val normalizer = new Normalizer()
	  .setInputCol("features")
	  .setOutputCol("norm_features")
	  .setP(2.0)  // L2 norm

	val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, word2Vec, normalizer))


    // 6. --- Assemble the Pipeline ---
//   val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf))

    // --- Time the main operations ---

    println("\nFitting the NLP pipeline...") // Fix: Ensure single-line string literal
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    println("\nTransforming data with the fitted pipeline...") // Fix: Ensure single-line string literal
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache() // Cache the result for efficiency
    val transformCount = transformedDF.count() // Force an action to trigger the transformation
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Calculate actual vocabulary size after tokenization and stop word removal
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1) // Filter out single-character tokens
      .distinct()
      .count()
    println(s"--> Actual vocabulary size after tokenization and stop word removal: $actualVocabSize unique terms.")

    // --- Show and Save Results ---
    println("\nSample of transformed data:") // Fix: Ensure single-line string literal
    transformedDF.select("text", "features").show(5, truncate = 50)

    val n_results = 20
    val results = transformedDF.select("text", "features").take(n_results)

    // 7. --- Write Metrics and Results to Separate Files ---

    // Write metrics to the log folder
    val log_path = "../log/lab17_metrics.log" // Corrected path
    new File(log_path).getParentFile.mkdirs() // Ensure directory exists
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics ---")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"HashingTF numFeatures set to: 20000")
      if (20000 < actualVocabSize) {
        logWriter.println(s"Note: numFeatures (20000) is smaller than actual vocabulary size ($actualVocabSize). Hash collisions are expected.")
      }
      logWriter.println(s"Metrics file generated at: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }

    // Write data results to the results folder
    val result_path = "../results/lab17_pipeline_output.txt" // Corrected path
    new File(result_path).getParentFile.mkdirs() // Ensure directory exists
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline Output (First $n_results results) ---")
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Word2Vec Vector: ${features.toString}")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      println(s"Successfully wrote $n_results results to $result_path")
    } finally {
      resultWriter.close()
    }
		
// Logistic Regression

    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

	val labeledDF = initialDF.withColumn("label", when(col("text").rlike("(?i)spark"), 1.0).otherwise(0.0))
	val Array(train, test) = labeledDF.randomSplit(Array(0.8, 0.2), seed = 1234L)

	val lr = new LogisticRegression()
	  .setLabelCol("label")
	  .setFeaturesCol("norm_features")
	  .setMaxIter(10)

	val pipelineLR = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, word2Vec, normalizer, lr))


    val modelLR = pipelineLR.fit(train)
    val predictions = modelLR.transform(test)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    println(s"--> Logistic Regression Accuracy = ${evaluator.evaluate(predictions)}")

  //COSINE SIMILARITY
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.Row

    // Hàm cosine similarity
    def cosineSimilarity(v1: Vector, v2: Vector): Double = {
      val dot = v1.toArray.zip(v2.toArray).map { case (x, y) => x * y }.sum
      val norm1 = math.sqrt(v1.toArray.map(x => x*x).sum)
      val norm2 = math.sqrt(v2.toArray.map(x => x*x).sum)
      if (norm1 == 0.0 || norm2 == 0.0) 0.0 else dot / (norm1 * norm2)
    }

    val queryRow: Row = transformedDF.select("norm_features", "text").collect()(0)
    val queryVec = queryRow.getAs[Vector]("norm_features")
    val queryText = queryRow.getAs[String]("text")

    println("\n================ QUERY DOCUMENT ================")
    println(queryText.take(200) + "...")
    println("================================================")

    val cosineUDF = udf((vec: Vector) => cosineSimilarity(queryVec, vec))

    val sims = transformedDF.withColumn("similarity", cosineUDF(col("norm_features")))

    println("\n================ TOP 10 SIMILAR DOCUMENTS ================")
    sims.orderBy(desc("similarity"))
      .select("similarity", "text")
      .show(10, truncate = 120)
    println("==========================================================")

    spark.stop()
    println("Spark Session stopped.")
  }
}