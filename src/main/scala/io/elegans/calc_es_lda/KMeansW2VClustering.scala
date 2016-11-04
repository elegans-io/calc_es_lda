package io.elegans.clustering

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors, Matrix}
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix, IndexedRowMatrix}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.elasticsearch.spark._
import scala.collection.mutable.ArrayBuffer
import scopt.OptionParser
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

/* import core nlp */
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
/* these are necessary since core nlp is a java library */
import java.util.Properties
import scala.collection.JavaConversions._

object KMeansW2VClustering {

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
    maxIterations: Int = 10,
    k: Int = 10
  )

  private def doClustering(params: Params) {
    val conf = new SparkConf().setAppName("W2V clustering")
    conf.set("es.nodes.wan.only", "true")
    conf.set("es.nodes", params.hostname)
    conf.set("es.port", params.port)

    val query: String = params.query
    conf.set("es.query", query)

    val sc = new SparkContext(conf)
    val search_res = sc.esRDD(params.search_path, "?q=*")

    val stopWords: Set[String] = params.stopwordFile match {
      case Some(stopwordFile) => sc.broadcast(scala.io.Source.fromFile(stopwordFile)
        .getLines().map(_.trim).toSet).value
      case None => Set.empty
    }

    /* docTerms: map of (docid, list_of_lemmas) */
    val used_fields = params.used_fields
    val docTerms = params.group_by_field match {
      case Some(group_by_field) =>
        val tmpDocTerms = search_res.map(s => {
          val key = s._2.getOrElse(group_by_field, "")
          (key, s._2)
        }
        ).groupByKey().map( s => {
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

    val w2vModel = Word2VecModel.load(sc, params.inputW2VModel)

    val v_size = w2vModel.getVectors.head._2.length

    /* docTermFreqs: mapping <doc_id> -> (vector_avg_of_term_vectors) */
    val docVectors = docTerms.filter(_._1 != None).filter(_._2 != List.empty[String]).map( e => {
      val v =  e._2.map(l => {
        try {
          w2vModel.transform(l).toDense.toArray
        } catch {
          case e: Exception => Vectors.zeros(v_size).toDense.toArray
        }
      })

      val v_sum = v.reduce((x,y) => Tuple2(x, y).zipped.map(_ + _))
      (e._1.toString, Vectors.dense(v_sum))
    })

  val numClusters = params.k
  val numIterations = params.maxIterations
  val indata = docVectors.map(_._2)
  val clusters = KMeans.train(indata, numClusters, numIterations)

  val predictions = docVectors.map(x => {(x._1, clusters.predict(x._2))})

  val WSSSE = clusters.computeCost(indata)
  println("Within Set Sum of Squared Errors (WSSSE) = " + WSSSE)

  predictions.saveAsTextFile(params.outputDir)
  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Clustering with ES data") {
      head("calculate clusters with data from an elasticsearch index using w2v representation of phrases.")
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
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[String]("inputW2VModel")
        .text(s"the input word2vec model")
        .action((x, c) => c.copy(inputW2VModel = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        doClustering(params)
      case _ =>
        sys.exit(1)
    }
  }
}
