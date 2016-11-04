package io.elegans.clustering

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

import scala.util.Try
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scopt.OptionParser

object W2VModelToSparkFormat {

  private case class Params(
      inputfile: String = "",
      outputdir: String = ""
  )

  private def convertModelToSparkW2V(inputfile: String, outputdir: String) {
    val conf = new SparkConf().setAppName("LDA from ES data with W2V")
    val sc = new SparkContext(conf)

    val fileName = Option(inputfile)
    val w2vfile = fileName match {
      case Some(trainedModelFN) =>
        scala.io.Source.fromFile(name=trainedModelFN, enc="UTF-8").getLines().map(_.trim)
      case None => List.empty
    }

    val model = w2vfile.map( line => {
      val items : Array[String] = line.split(" ")
      val key : String = items(0)
      val values : Array[Float] = items.drop(1).map(x => Try(x.toFloat).getOrElse(0.toFloat))
      (key, values)
    })

    val w2vModel = new Word2VecModel(model.toMap)

    w2vModel.save(sc, outputdir)
  }

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("W2VModelToSparkFormat") {
      head("Load word2vec model in textual format separated by spaces (<term> <v0> .. <vn>) and save it in spark format.")
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
