package io.elegans.clustering

import scala.util.Try
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scopt.OptionParser
import org.apache.spark.storage.StorageLevel

object ReduceW2VModel {

  lazy val textProcessingUtils = new TextProcessingUtils /* lazy initialization of TextProcessingUtils class */
  lazy val loadData = new LoadData

  private case class Params(
    inputfile: Option[String] = None,
    hostname: String = "localhost",
    port: String = "9200",
    search_path: String = "jenny-en-0/question",
    query: String = """{ "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }""",
    used_fields: Seq[String] = Seq[String]("question", "answer"),
    inputmodel: String = "",
    group_by_field: Option[String] = None,
    outputfile: String = "/tmp/w2v_model.txt",
    stopwordsFile: Option[String] = None
  )

  private def doReduceW2V(params: Params) {
    val conf : SparkConf = new SparkConf().setAppName("ReduceW2VModel")

    if (params.inputfile.isEmpty) {
      conf.set("es.nodes.wan.only", "true")
      conf.set("es.nodes", params.hostname)
      conf.set("es.port", params.port)
      val query: String = params.query
      conf.set("es.query", query)
    }

    val sc : SparkContext = new SparkContext(conf)

    val stopWords = params.stopwordsFile match {  /* check the stopWord variable */
      case Some(stopwordsFile) => sc.broadcast(scala.io.Source.fromFile(stopwordsFile) /* load the stopWords if Option
                                                variable contains an existing value */
        .getLines().map(_.trim).toSet)
      case None => sc.broadcast(Set.empty[String]) /* set an empty string if Option variable is None */
    }

    val docTerms = if (params.inputfile.isEmpty) {
      val documentTerms = loadData.loadDocumentsFromES(sc = sc, search_path = params.search_path,
        used_fields = params.used_fields, group_by_field = params.group_by_field).mapValues(x => {
        textProcessingUtils.tokenizeSentence(x, stopWords, 0).toSet
      })
      documentTerms
    } else {
      val documentTerms = loadData.loadDocumentsFromFile(sc = sc, input_path = params.inputfile.get).mapValues(x => {
        textProcessingUtils.tokenizeSentence(x, stopWords, 0).toSet
      })
      documentTerms
    }

    docTerms.persist(StorageLevel.MEMORY_AND_DISK_SER_2)

    val terms : Set[String] = docTerms.map(_._2).fold(Set.empty[String])((a, b) => a ++ b)

    val w2vfile = sc.textFile(params.inputmodel).map(_.trim)

    val filteredEntries = w2vfile.map( line => {
      val items : Array[String] = line.split(" ")
      val key : String = items(0)
      val values : Array[Double] = items.drop(1).map(x => Try(x.toDouble).getOrElse(0.toDouble))
      (key, values)
    }).filter(x => terms(x._1))

    filteredEntries.map(x => x._1 + " " + x._2.mkString(" ")).saveAsTextFile(params.outputfile)

  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Generate a reduced W2V model by selecting only the vectors of the words used in the dataset") {
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
      opt[String]("inputmodel")
        .text(s"the file with the model")
        .action((x, c) => c.copy(inputmodel = x))
      opt[String]("outputfile")
        .text(s"the output file" +
          s"  default: ${defaultParams.outputfile}")
        .action((x, c) => c.copy(outputfile = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        doReduceW2V(params)
      case _ =>
        sys.exit(1)
    }
  }
}
