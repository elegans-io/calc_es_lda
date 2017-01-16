package io.elegans.clustering

import org.apache.spark.mllib.feature.{Word2Vec}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.elasticsearch.spark._
import scopt.OptionParser

object TrainW2V {

  lazy val textProcessingUtils = new TextProcessingUtils /* lazy initialization of TextProcessingUtils class */
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
    stopwordsFile: Option[String] = Option("stopwords/en_stopwords.txt"),
    vector_size: Int = 300,
    word_window_size: Int = 5,
    learningRate: Double = 0.025,
    strInBase64: Boolean = false
  )

  private def doTrainW2V(params: Params) {
    val conf = new SparkConf().setAppName("LDA from ES data")

    if (params.inputfile.isEmpty) {
      conf.set("es.nodes.wan.only", "true")
      conf.set("es.index.read.missing.as.empty", "yes")
      conf.set("es.nodes", params.hostname)
      conf.set("es.port", params.port)
      val query: String = params.query
      conf.set("es.query", query)
    }

    val sc = new SparkContext(conf)
    val search_res = sc.esRDD(params.search_path, "?q=*")

    val stopWords = params.stopwordsFile match {  /* check the stopWord variable */
      case Some(stopwordsFile) => sc.broadcast(scala.io.Source.fromFile(stopwordsFile) /* load the stopWords if Option
                                                variable contains an existing value */
        .getLines().map(_.trim).toSet)
      case None => sc.broadcast(Set.empty[String]) /* set an empty string if Option variable is None */
    }

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

    val word2vec = new Word2Vec()
    word2vec.setVectorSize(params.vector_size)
    word2vec.setWindowSize(params.word_window_size)
    word2vec.setLearningRate(params.learningRate)

    val model = word2vec.fit(docTerms.map(_._2))

    // Save and load model
    model.save(sc, params.outputDir)
  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Train a W2V model") {
      head("Train a W2V model taking input data from ES.")
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
      opt[Int]("vector_size")
        .text(s"the vector size" +
          s"  default: ${defaultParams.vector_size}")
        .action((x, c) => c.copy(vector_size = x))
      opt[Int]("word_window_size")
        .text(s"the word window size" +
          s"  default: ${defaultParams.word_window_size}")
        .action((x, c) => c.copy(word_window_size = x))
      opt[Double]("learningRate")
        .text(s"the learningRate" +
          s"  default: ${defaultParams.learningRate}")
        .action((x, c) => c.copy(learningRate = x))
      opt[Unit]("strInBase64")
        .text(s"specify if the text is encoded in base64 (only supported by loading from file)" +
          s"  default: ${defaultParams.strInBase64}")
        .action((x, c) => c.copy(strInBase64 = true))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        doTrainW2V(params)
      case _ =>
        sys.exit(1)
    }
  }
}
