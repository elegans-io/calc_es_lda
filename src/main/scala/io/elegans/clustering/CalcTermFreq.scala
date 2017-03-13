package io.elegans.clustering

import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel, EMLDAOptimizer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.mutable.HashMap
import scala.collection.mutable.MutableList

import scopt.OptionParser

import org.apache.spark.storage.StorageLevel

object CalcTermFreq {

  lazy val textProcessingUtils = new TextProcessingUtils /* lazy initialization of TextProcessingUtils class */
  lazy val loadData = new LoadData

  private case class Params(
    inputfile: Option[String] = None,
    outputDir: String = "/tmp/calc_freq_output",
    levels: Int = 3,
    numpartitions: Int = 1000,
    appname: String = "CalcTerm freq",
    minfreq: Int = 0
  )

  private def doCalcTermFreq(params: Params) {
    val conf = new SparkConf().setAppName(params.appname)
    val sc = new SparkContext(conf)

    val stopWords = sc.broadcast(Set.empty[String]) /* set an empty string if Option variable is None */

    val path = params.inputfile.get + ("/*" * params.levels)
    val files = sc.wholeTextFiles(path=path, minPartitions=params.numpartitions)

    files.persist(StorageLevel.MEMORY_AND_DISK)

    val termCount = files.flatMap(x => {
      val token_list = textProcessingUtils.tokenizeSentence(x._2, stopWords, 0)
      token_list
    }).map(x => (x, 1)).reduceByKey((a, b) => {a + b})

    termCount.persist(StorageLevel.MEMORY_AND_DISK)

    val tot_terms : Int = termCount.values.fold(0)((a, b) => {a + b})
    val tot_files = files.count

    val total_words_h_element = "words: " + tot_terms
    val line_split = "---------------------------------------------"

    val header = sc.parallelize(List(total_words_h_element, line_split))

    val minfreq = params.minfreq
    val term_out = termCount.map(x => (x._1, x._2, (x._2 : Double)/(tot_terms : Double))).filter(x => x._2 >= minfreq)
      .sortBy(x => x._2, ascending = false)
    val outResultsDirnameFilePath = params.outputDir
    val wordlist = term_out.map(x => x._1 + "\t" + x._2 + "\t(" + x._3 * 1000 + " ‰)")
    val output_data = (header ++ wordlist)
    output_data.saveAsTextFile(outResultsDirnameFilePath)
  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Tokenize terms and count term frequency from a set of files") {
      head("calculate term frequency.")
      help("help").text("prints this usage text")
      opt[String]("inputfile")
        .text(s"the path e.g. tmp/dir" +
          s"  default: ${defaultParams.inputfile}")
        .action((x, c) => c.copy(inputfile = Option(x)))
      opt[String]("appname")
        .text(s"application name" +
          s"  default: ${defaultParams.appname}")
        .action((x, c) => c.copy(appname = x))
      opt[String]("outputDir")
        .text(s"the where to store the output files: topics and document per topics" +
          s"  default: ${defaultParams.outputDir}")
        .action((x, c) => c.copy(outputDir = x))
      opt[Int]("levels")
        .text(s"the levels where to search for files e.g. for level=2 => tmp/dir/*/*" +
          s"  default: ${defaultParams.levels}")
        .action((x, c) => c.copy(levels = x))
      opt[Int]("numpartitions")
        .text(s"the number of partitions reading files" +
          s"  default: ${defaultParams.numpartitions}")
        .action((x, c) => c.copy(numpartitions = x))
      opt[Int]("minfreq")
        .text(s"remove words with a frequency less that minfreq" +
          s"  default: ${defaultParams.minfreq}")
        .action((x, c) => c.copy(minfreq = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        doCalcTermFreq(params)
      case _ =>
        sys.exit(1)
    }
  }
}
