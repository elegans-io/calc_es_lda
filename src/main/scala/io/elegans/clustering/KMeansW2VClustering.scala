package io.elegans.clustering

import org.apache.spark.mllib.feature.{Word2VecModel}
import org.apache.spark.mllib.linalg.{Vectors}
import org.apache.spark.mllib.clustering.{KMeans}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.elasticsearch.spark._

import scopt.OptionParser
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming._

object KMeansW2VClustering {

  lazy val textProcessingUtils = new TextProcessingUtils /* lazy initialization of TextProcessingUtils class */

  private case class Params(
    hostname: String = "localhost",
    port: String = "9200",
    search_path: String = "jenny-en-0/question",
    query: String = """{ "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }""",
    used_fields: Seq[String] = Seq[String]("question", "answer"),
    group_by_field: Option[String] = None,
    outputDir: String = "/tmp",
    stopwordsFile: Option[String] = None,
    inputW2VModel: String = "",
    maxIterations: Int = 1000,
    max_k: Int = 10,
    min_k: Int = 8,
    avg: Boolean = false,
    tfidf: Boolean = false,
    scale: Boolean = false
  )

  private def scaleTo1(v: Vector) : Vector = {
    val array = v.toDense.toArray
    val norm = math.sqrt(array.map(math.pow(_,2)).sum)
    val scaled_v = array.map(x => { x / norm } )
    Vectors.dense(scaled_v)
  }

  private def doClustering(params: Params) {
    val conf = new SparkConf().setAppName("W2V clustering")
    conf.set("es.nodes.wan.only", "true")
    conf.set("es.nodes", params.hostname)
    conf.set("es.port", params.port)

    val query: String = params.query
    conf.set("es.query", query)

    val sc = new SparkContext(conf)
    val search_res = sc.esRDD(params.search_path, "?q=*")

    val stopWords = params.stopwordsFile match {  /* check the stopWord variable */
      case Some(stopwordsFile) => sc.broadcast(scala.io.Source.fromFile(stopwordsFile) /* load the stopWords if Option
                                                variable contains an existing value */
        .getLines().map(_.trim).toSet)
      case None => sc.broadcast(Set.empty[String]) /* set an empty string if Option variable is None */
    }

    /* docTerms: map of (docid, list_of_lemmas) */
    val used_fields = params.used_fields
    val docTerms = params.group_by_field match {
      case Some(group_by_field) =>
        val tmpDocTerms = search_res.map(s => {
          val key = s._2.getOrElse(group_by_field, "")
          (key, List(s._2))
        }
        ).reduceByKey(_ ++ _).map( s => {
          val conversation : String = s._2.foldRight("")((a, b) =>
            try {
              val c = used_fields.map( v => { a.getOrElse(v, None) } )
                .filter(x => x != None).mkString(" ") + " " + b
              c
            } catch {
              case e: Exception => ""
            }
          )
          try {
            val token_list = textProcessingUtils.tokenizeSentence(conversation, stopWords)
            val doc_lemmas : Tuple2[String, List[String]] =
              (s._1.toString, token_list)
            doc_lemmas
          } catch {
            case e: Exception => Tuple2(None, List.empty[String])
          }
        })
        tmpDocTerms
      case None =>
        val tmpDocTerms = search_res.map(s => {
          try {
            val doctext = used_fields.map( v => {
              s._2.getOrElse(v, "")
            } ).mkString(" ")
            val token_list = textProcessingUtils.tokenizeSentence(doctext, stopWords)
            val doc_lemmas : Tuple2[String, List[String]] =
              (s._1.toString, token_list)
            doc_lemmas
          } catch {
            case e: Exception => Tuple2(None, List.empty[String])
          }
        })
        tmpDocTerms
    }

    val documents = docTerms.filter(_._1 != "").filter(_._2 != List.empty[String]) //.filter(_._2 != List("none"))

    val w2vModel = Word2VecModel.load(sc, params.inputW2VModel)
    val v_size = w2vModel.getVectors.head._2.length

    /* docTermFreqs: mapping <doc_id> -> (vector_avg_of_term_vectors) */
    val docVectors = if(params.tfidf) {
      val hashingTF = new HashingTF()
      val tf: RDD[Vector] = hashingTF.transform(documents.values) // values -> get the value from the (key, value) pair
      tf.cache()

      val idf_filtered = new IDF(minDocFreq = 2).fit(tf) // compute the inverse document frequency
      val tfidf: RDD[Vector] = idf_filtered.transform(tf) // transforms term frequency (TF) vectors to TF-IDF vectors

      val documents_idf = documents.zip(tfidf)

      val sentenceVectors = documents_idf.map(e => {
        val term_tfidf_weight = e._2
        val v = e._1._2.map(lemma => {
          try {
            val vector = w2vModel.transform(lemma).toDense.toArray //transform word to vector
            val lemma_hash_id = hashingTF.transform(Seq(lemma))
            val max_index = lemma_hash_id.argmax
            val term_weight = term_tfidf_weight(max_index)
            val weighted_vector = vector.map(item => {
              item * term_weight
            }).toVector.toArray
            weighted_vector
          } catch {
            case e: Exception => Vectors.zeros(v_size).toDense.toArray
          }
        }).filter(! _.sameElements(Vectors.zeros(v_size).toDense.toArray))

        val v_phrase_sum = v.fold(Vectors.zeros(v_size).toDense.toArray)((x, y) => Tuple2(x, y).zipped.map(_ + _))
        if (params.avg) {
          val normal: Array[Double] = Array.fill[Double](v_size)(v_size)
          val v_phrase = List(v_phrase_sum, normal).reduce((x, y) => Tuple2(x, y).zipped.map(_ / _))
          (e._1._1.toString, Vectors.dense(v_phrase))
        } else {
          (e._1._1.toString, Vectors.dense(v_phrase_sum))
        }
      })
      sentenceVectors
    } else {
      val sentenceVectors = documents.map(e => {
        val v = e._2.map(lemma => {
          try {
            val vector = w2vModel.transform(lemma).toDense.toArray //transform word to vector
            vector
          } catch {
            case e: Exception => Vectors.zeros(v_size).toDense.toArray
          }
        }).filter(! _.sameElements(Vectors.zeros(v_size).toDense.toArray))

        val v_phrase_sum = v.fold(Vectors.zeros(v_size).toDense.toArray)((x, y) => Tuple2(x, y).zipped.map(_ + _))
        if (params.avg) {
          val normal: Array[Double] = Array.fill[Double](v_size)(v_size)
          val v_phrase = List(v_phrase_sum, normal).reduce((x, y) => Tuple2(x, y).zipped.map(_ / _))
          (e._1.toString, Vectors.dense(v_phrase))
        } else {
          (e._1.toString, Vectors.dense(v_phrase_sum))
        }
      })

      sentenceVectors
    }

  val scaler : Vector => Vector = if(params.scale) { scaleTo1 } else { identity }
  val documentVectors = docVectors.map(x => {(x._1, scaler(x._2))})
  val indata = documentVectors.map(_._2)
  indata.cache()

  val max_k : Int = params.max_k
  val min_k : Int = params.min_k
  var k : Int = min_k
  val numIterations = params.maxIterations
  var iterate : Boolean = true
  do {
    val clusters = KMeans.train(indata, k, numIterations)

    val predictions = documentVectors.map(x => {
      (x._1, clusters.predict(x._2))
    })

    val WSSSE = clusters.computeCost(indata)
    println("Within Set Sum of Squared Errors K(" + k + ") (WSSSE) = " + WSSSE)

    val outTopicPerDocumentDirname = "KMEANS_W2V_TOPICSxDOC_K." + k
    val outTopicPerDocumentFilePath = params.outputDir + "/" + outTopicPerDocumentDirname

    predictions.saveAsTextFile(outTopicPerDocumentFilePath)
    k += 1
    iterate = if (k < max_k) true else false
  } while (iterate)

  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Clustering with ES data") {
      head("calculate clusters with data from an elasticsearch index using w2v representation of phrases.")
      help("help").text("prints this usage text")
      opt[String]("hostname")
        .text(s"the hostname of the elasticsearch instance" +
          s"  default: ${defaultParams.hostname}")
        .action((x, c) => c.copy(hostname = x))
      opt[String]("port")
        .text(s"the port of the elasticsearch instance" +
          s"  default: ${defaultParams.port}")
        .action((x, c) => c.copy(port = x))
      opt[String]("group_by_field")
        .text(s"group the search results by field e.g. conversation, None => no grouping" +
          s"  default: ${defaultParams.group_by_field}")
        .action((x, c) => c.copy(group_by_field = Option(x)))
      opt[String]("search_path")
        .text(s"the search path on elasticsearch e.g. <index name>/<type name>" +
          s"  default: ${defaultParams.search_path}")
        .action((x, c) => c.copy(search_path = x))
      opt[String]("query")
        .text(s"a json string with the query" +
          s"  default: ${defaultParams.query}")
        .action((x, c) => c.copy(query = x))
      opt[String]("stopwordsFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
          s"  default: ${defaultParams.stopwordsFile}")
        .action((x, c) => c.copy(stopwordsFile = Option(x)))
      opt[Seq[String]]("used_fields")
        .text(s"list of fields to use for LDA, if more than one they will be merged" +
          s"  default: ${defaultParams.used_fields}")
        .action((x, c) => c.copy(used_fields = x))
      opt[String]("outputDir")
        .text(s"the where to store the output files: topics and document per topics" +
          s"  default: ${defaultParams.outputDir}")
        .action((x, c) => c.copy(outputDir = x))
      opt[Int]("min_k")
        .text(s"min number of topics. default: ${defaultParams.min_k}")
        .action((x, c) => c.copy(min_k = x))
      opt[Int]("max_k")
        .text(s"max number of topics. default: ${defaultParams.max_k}")
        .action((x, c) => c.copy(max_k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[String]("inputW2VModel")
        .text(s"the input word2vec model")
        .action((x, c) => c.copy(inputW2VModel = x))
      opt[Unit]("avg").text("this flag enable the vectors")
        .action( (x, c) => c.copy(avg = true))
      opt[Unit]("tfidf").text("this flag enable tfidf term weighting")
        .action( (x, c) => c.copy(tfidf = true))
      opt[Unit]("scale").text("this flag enable vector scaling to 1")
        .action( (x, c) => c.copy(scale = true))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        doClustering(params)
      case _ =>
        sys.exit(1)
    }
  }
}
