package io.elegans.clustering

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Matrix, Vector, Vectors}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.{IndexedRowMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.elasticsearch.spark._

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scopt.OptionParser
import org.apache.spark.rdd.RDD
import org.apache.spark._
import org.apache.spark.streaming._

/* import core nlp */
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
/* these are necessary since core nlp is a java library */
import java.util.Properties
import scala.collection.JavaConversions._

/**
  * Created by angelo on 14/11/16.
  */
object GetW2VSimilarSentences {

  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.setProperty("annotators", "tokenize, ssplit, pos, lemma")
    val pipeline: StanfordCoreNLP = new StanfordCoreNLP(props)
    pipeline
  }

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }

  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP): List[String] = {
    val doc: Annotation = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences;
         token <- sentence.get(classOf[TokensAnnotation])) {
      val lemma = token.getString(classOf[LemmaAnnotation])
      val lc_lemma = lemma.toLowerCase
      if (lc_lemma.length > 2 && !stopWords.contains(lc_lemma) && isOnlyLetters(lc_lemma)) {
        lemmas += lc_lemma.toLowerCase
      }
    }
    lemmas.toList
  }

  def cosineSimilarity(a: Vector, b: Vector) : Double = {
    val values = a.toDense.toArray.zip(b.toDense.toArray).map(x => {
      (x._1 * x._2, scala.math.pow(x._1, 2), scala.math.pow(x._2, 2))
      }).reduce((a,b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
    val cs = values._1 / (scala.math.sqrt(values._2) * scala.math.sqrt(values._3))
    cs
  }

  private case class Params(
                             hostname: String = "localhost",
                             port: String = "9200",
                             search_path: String = "jenny-en-0/question",
                             query: String = """{ "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }""",
                             used_fields: Seq[String] = Seq[String]("question", "answer"),
                             group_by_field: Option[String] = None,
                             outputDir: String = "/tmp",
                             stopwordFile: Option[String] = Option("stopwords/en_stopwords.txt"),
                             inputW2VModel: String = "",
                             avg: Boolean = true,
                             tfidf : Boolean = false,
                             input_sentences : String = "",
                             similarity_threshold : Double = 0.0
                           )

  private def doClustering(params: Params) {
    val conf = new SparkConf().setAppName("W2V clustering")
    conf.set("es.nodes.wan.only", "true")
    conf.set("es.nodes", params.hostname)
    conf.set("es.port", params.port)

    val query: String = params.query
    conf.set("es.query", query)

    val sc = new SparkContext(conf)
    val ssc = new StreamingContext(sc, Seconds(1))
    val search_res = sc.esRDD(params.search_path, "?q=*")

    val stopWords: Set[String] = params.stopwordFile match {
      case Some(stopwordFile) => sc.broadcast(scala.io.Source.fromFile(stopwordFile)
        .getLines().map(_.trim).toSet).value
      case None => Set.empty
    }

    val querySentences = sc.textFile(params.input_sentences).map(_.trim)

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
            val pipeline = createNLPPipeline()
            val doc_lemmas : Tuple2[String, List[String]] =
              (s._1.toString, plainTextToLemmas(conversation, stopWords, pipeline))
            doc_lemmas
          } catch {
            case e: Exception => Tuple2(None, List.empty[String])
          }
        })
        tmpDocTerms
      case None =>
        val tmpDocTerms = search_res.map(s => {
          try {
            val pipeline = createNLPPipeline()
            val doctext = used_fields.map( v => {
              s._2.getOrElse(v, "")
            } ).mkString(" ")
            val doc_lemmas : Tuple2[String, List[String]] =
              (s._1.toString, plainTextToLemmas(doctext, stopWords, pipeline))
            doc_lemmas
          } catch {
            case e: Exception => Tuple2(None, List.empty[String])
          }
        })
        tmpDocTerms
    }

    val documents = docTerms.filter(_._1 != "").filter(_._2 != List.empty[String]).map(x => (x._1.toString, x._2))
    val w2vModel = Word2VecModel.load(sc, params.inputW2VModel)
    val query_id_prefix : String = "io.elegans.clustering.query_sentence_tmp_"
    val v_size = w2vModel.getVectors.head._2.length

    val enumeratedQueryItems = querySentences.zipWithIndex.map(x => {
      val pipeline = createNLPPipeline()
      (query_id_prefix + x._2,
        plainTextToLemmas(x._1, stopWords, pipeline)
        )
    })

    val merged_collection = enumeratedQueryItems.union(documents)

    /* docTermFreqs: mapping <doc_id> -> (vector_avg_of_term_vectors) */
    val docVectors = if(params.tfidf) {
      val hashingTF = new HashingTF()
      val tf: RDD[Vector] = hashingTF.transform(merged_collection.values)
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
          (e._1._1.toString, Vectors.dense(v_phrase))
        } else {
          (e._1._1.toString, Vectors.dense(v_phrase_sum))
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
          (e._1.toString, Vectors.dense(v_phrase))
        } else {
          (e._1.toString, Vectors.dense(v_phrase_sum))
        }
      })
      sentenceVectors
    }

    val numOfQuery : Long = enumeratedQueryItems.count()
    val queryVectors = sc.parallelize(docVectors.take(numOfQuery.toInt))

    val similarity_values = queryVectors.cartesian(docVectors).map(x => {
      val cs = cosineSimilarity(x._1._2, x._2._2)
      (x._1._1, x._2._1, cs)
    }).filter(_._3 >= params.similarity_threshold)

    val outResultsDirnameFilePath = params.outputDir
    similarity_values.saveAsTextFile(outResultsDirnameFilePath)

  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Search similar sentences") {
      head("perform a similarity search using cosine vector as distance function")
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
      opt[String]("stopwordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
          s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = Option(x)))
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
      opt[Unit]("avg").text("this flag disable the vectors")
        .action( (x, c) => c.copy(avg = false))
      opt[Unit]("tfidf").text("this flag enable tfidf term weighting")
        .action( (x, c) => c.copy(tfidf = false))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        doClustering(params)
      case _ =>
        sys.exit(1)
    }
  }
}

