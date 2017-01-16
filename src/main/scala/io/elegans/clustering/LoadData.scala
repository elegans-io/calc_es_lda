package io.elegans.clustering

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.elasticsearch.spark._
import org.apache.commons.codec.binary.Base64
import org.apache.commons.math3.analysis.function.Identity

class LoadData {

  def loadDocumentsFromFile(sc: SparkContext, input_path: String): RDD[(String, String)] = {
    val inputfile = sc.textFile(input_path).map(_.trim) /* read the input file in textual format */
    /* process the lines of the input file, assigning to each line a progressive ID*/

    val tokenizedSentences = inputfile.zipWithIndex.map( line => {
      val original_string = line._1 /* the original string as is on file */
      val id : String = line._2.toString /* the unique sentence id converted to string */
      (id, original_string)
    })
    tokenizedSentences
  }

  def loadDocumentsFromFileBase64(sc: SparkContext, input_path: String): RDD[(String, String)] = {
    val inputfile = sc.textFile(input_path).map(_.trim) /* read the input file in textual format */
    /* process the lines of the input file, assigning to each line a progressive ID*/

    val tokenizedSentences = inputfile.zipWithIndex.map( line => {
      val bytes: Array[Byte] = line._1.getBytes
      val decoded : Array[Byte] = Base64.decodeBase64(bytes)
      val original_string = new String(decoded, "utf-8") /* the original string as is on file */
      val id : String = line._2.toString /* the unique sentence id converted to string */
      (id, original_string)
    })
    tokenizedSentences
  }

  def loadDocumentsFromES(sc: SparkContext, search_path: String, used_fields: Seq[String],
                          group_by_field: Option[String]
                         ): RDD[(String, String)] = {
    val search_res = sc.esRDD(search_path, "?q=*")
    /* docTerms: map of (docid, document) */
    val documents = group_by_field match {
      case Some(group_by_field) =>
        val tmpDocTerms = search_res.map(s => {
          val key = s._2.getOrElse(group_by_field, "")
          (key.toString, List(s._2))
        }).reduceByKey(_ ++ _).map( s => {
          val conversation: String = s._2.foldRight("")((a, b) =>
            try {
              val components = used_fields.map(v => {
                a.getOrElse(v, "")
              })
                .filter(x => x != None).mkString(" ") + " " + b
              components
            } catch {
              case e: Exception => ""
            }
          )
          val docs = (s._1, conversation)
          docs
        })
        tmpDocTerms
      case None =>
        val tmpDocTerms = search_res.map(s => {
          try {
            val doctext = used_fields.map( v => {
              s._2.getOrElse(v, "")
            } ).mkString(" ")
            val docs = (s._1, doctext)
            docs
          } catch {
            case e: Exception => ("", "")
          }
        })
        tmpDocTerms
    }
    documents
  }

}
