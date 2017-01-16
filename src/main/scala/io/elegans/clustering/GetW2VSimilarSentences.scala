package io.elegans.clustering

import org.apache.spark.mllib.feature.{Word2VecModel}
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scopt.OptionParser
import org.apache.spark.rdd.RDD

/**
  * Created by angelo on 14/11/16.
  */
object GetW2VSimilarSentences {

  lazy val textProcessingUtils = new TextProcessingUtils /* lazy initialization of TextProcessingUtils class */
  lazy val termVectors = new TermVectors
  lazy val loadData = new LoadData

  private case class Params(
    inputfile: Option[String] = None,
    hostname: String = "localhost",
    port: String = "9200",
    search_path: String = "jenny-en-0/question",
    query: String = """{ "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }""",
    used_fields: Seq[String] = Seq[String]("question", "answer"),
    group_by_field: Option[String] = None,
    outputDir: String = "/tmp",
    stopwordsFile: Option[String] = None,
    inputW2VModel: String = "",
    avg: Boolean = false,
    tfidf : Boolean = false,
    input_sentences : String = "",
    similarity_threshold : Double = 0.0,
    strInBase64: Boolean = false
  )

  private def doClustering(params: Params) {
    val conf = new SparkConf().setAppName("W2V clustering").set("spark.driver.maxResultSize", "16g")

    if (params.inputfile.isEmpty) {
      conf.set("es.nodes.wan.only", "true")
      conf.set("es.index.read.missing.as.empty", "yes")
      conf.set("es.nodes", params.hostname)
      conf.set("es.port", params.port)
      val query: String = params.query
      conf.set("es.query", query)
    }

    val sc = new SparkContext(conf)

    val stopWords = params.stopwordsFile match {  /* check the stopWord variable */
      case Some(stopwordsFile) => sc.broadcast(scala.io.Source.fromFile(stopwordsFile) /* load the stopWords if Option
                                                variable contains an existing value */
        .getLines().map(_.trim).toSet)
      case None => sc.broadcast(Set.empty[String]) /* set an empty string if Option variable is None */
    }

    val querySentences = sc.textFile(params.input_sentences).map(_.trim)

    val docTerms = if (params.inputfile.isEmpty) {
      val documentTerms = loadData.loadDocumentsFromES(sc = sc, search_path = params.search_path,
        used_fields = params.used_fields, group_by_field = params.group_by_field).mapValues(x => {
        textProcessingUtils.tokenizeSentence(x, stopWords, 0)
      })
      documentTerms
    } else {
      val documentTerms = params.strInBase64 match {
        case false => loadData.loadDocumentsFromFile(sc = sc, input_path = params.inputfile.get).mapValues(x => {
          textProcessingUtils.tokenizeSentence(x, stopWords, 0)
        })
        case true => loadData.loadDocumentsFromFileBase64(sc = sc, input_path = params.inputfile.get).mapValues(x => {
          textProcessingUtils.tokenizeSentence(x, stopWords, 0)
        })
      }
      documentTerms
    }

    val documents = docTerms.filter(_._1 != "").filter(_._2 != List.empty[String]).map(x => (x._1.toString, x._2))

    val w2vModel = Word2VecModel.load(sc, params.inputW2VModel)
    val query_id_prefix : String = "io.elegans.clustering.query_sentence_tmp_"
    val v_size = w2vModel.getVectors.head._2.length

    val enumeratedQueryItems = querySentences.zipWithIndex.map(x => {
      val token_list = textProcessingUtils.tokenizeSentence(x._1, stopWords, 0)
      (query_id_prefix + x._2, token_list)
    })

    val merged_collection = enumeratedQueryItems.union(documents)

    /* docTermFreqs: mapping <doc_id> -> (vector_avg_of_term_vectors) */
    val docVectors = if(params.tfidf) {
      val hashingTF = new HashingTF()
      val tf: RDD[Vector] = hashingTF.transform(merged_collection.values) // values -> get the value from the (key, value) pair
      tf.cache()

      val idf_filtered = new IDF(minDocFreq = 2).fit(tf) // compute the inverse document frequency
      val tfidf: RDD[Vector] = idf_filtered.transform(tf) // transforms term frequency (TF) vectors to TF-IDF vectors

      val documents_idf = merged_collection.zip(tfidf)

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
          (e._1._1.toString, Vectors.dense(v_phrase).toSparse)
        } else {
          (e._1._1.toString, Vectors.dense(v_phrase_sum).toSparse)
        }
      })
      sentenceVectors
    } else {
      val sentenceVectors = merged_collection.map(e => {
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
          (e._1.toString, Vectors.dense(v_phrase).toSparse)
        } else {
          (e._1.toString, Vectors.dense(v_phrase_sum).toSparse)
        }
      })
      sentenceVectors
    }

    val numOfQuery : Long = enumeratedQueryItems.count()
    val queryVectors = sc.parallelize(docVectors.take(numOfQuery.toInt))

    val similarity_values = queryVectors.cartesian(docVectors).filter(x => {x._1._1 != x._2._1}).map(x => {
      val cs = termVectors.cosineSimilarity(x._1._2, x._2._2)
      (cs, x._1._1, x._2._1)
    }).filter(_._1 >= params.similarity_threshold)

    val outResultsDirnameFilePath = params.outputDir
    similarity_values.saveAsTextFile(outResultsDirnameFilePath)
  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Search similar sentences") {
      head("perform a similarity search using cosine vector as distance function")
      help("help").text("prints this usage text")
      opt[String]("inputfile")
        .text(s"the file with sentences (one per line), when specified elasticsearch is not used" +
          s"  default: ${defaultParams.inputfile}")
        .action((x, c) => c.copy(inputfile = Option(x)))
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
      opt[String]("inputW2VModel")
        .text(s"the input word2vec model")
        .action((x, c) => c.copy(inputW2VModel = x))
      opt[String]("input_sentences")
        .text(s"the list of input sentences" + "  default: ${defaultParams.input_sentences}")
        .action((x, c) => c.copy(input_sentences = x))
      opt[Double]("similarity_threshold")
        .text(s"cutoff threshold" + "  default: ${defaultParams.similarity_threshold}")
        .action((x, c) => c.copy(similarity_threshold = x))
      opt[Unit]("avg").text("this flag enable the vectors")
        .action( (x, c) => c.copy(avg = true))
      opt[Unit]("tfidf").text("this flag enable tfidf term weighting")
        .action( (x, c) => c.copy(tfidf = true))
      opt[Unit]("strInBase64")
        .text(s"specify if the text is encoded in base64 (only supported by loading from file)" +
          s"  default: ${defaultParams.strInBase64}")
        .action((x, c) => c.copy(strInBase64 = true))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        doClustering(params)
      case _ =>
        sys.exit(1)
    }
  }
}

