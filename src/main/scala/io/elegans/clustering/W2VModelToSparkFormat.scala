package io.elegans.clustering

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD

import scala.util.Try
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import scopt.OptionParser

object W2VModelToSparkFormat {

  private case class Params(
      inputfile: String = "",
      outputdir: String = ""
  )

  private def convertModelToSparkW2V(inputfile: String, outputdir: String) {
    val conf = new SparkConf().setAppName("LDA from ES data with W2V")
      .set("spark.driver.maxResultSize", "16g")

    val sc = new SparkContext(conf)

    val w2vfile = sc.textFile(inputfile).map(_.trim)

    val model = w2vfile.map( line => {
      val items : Array[String] = line.split(" ")
      val key : String = items(0)
      val values : Array[Float] = items.drop(1).map(x => Try(x.toFloat).getOrElse(0.toFloat))
      (key, values)
    })

    model.persist(StorageLevel.MEMORY_AND_DISK)
    val w2vModel = new Word2VecModel(model.collectAsMap().toMap)

    w2vModel.save(sc, outputdir)
  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("W2VModelToSparkFormat") {
      head("Load word2vec model in textual format separated by spaces (<term> <v0> .. <vn>) and save it in spark format.")
      help("help").text("prints this usage text")
      opt[String]("inputfile")
        .text(s"the file with the model")
        .action((x, c) => c.copy(inputfile = x))
      opt[String]("outputdir")
        .text(s"the port of the elasticsearch instance")
        .action((x, c) => c.copy(outputdir = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        convertModelToSparkW2V(params.inputfile, params.outputdir)
      case _ =>
        sys.exit(1)
    }
  }
}
