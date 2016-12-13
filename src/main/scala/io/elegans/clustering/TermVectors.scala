package io.elegans.clustering

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}

class TermVectors {

  /** Calculate the cosine similarity between two vectors
    *
    * @param first_vector first vector
    * @param second_vector second vector
    * @return the cosine between the two input vectors
    */
  def cosineSimilarity(first_vector: SparseVector, second_vector: SparseVector) : Double = {
    val values = first_vector.toDense.toArray.zip(second_vector.toDense.toArray).map(x => {
      (x._1 * x._2, scala.math.pow(x._1, 2), scala.math.pow(x._2, 2))
    }).reduce((vector_A, vector_B) =>
      (vector_A._1 + vector_B._1, vector_A._2 + vector_B._2, vector_A._3 + vector_B._3))
    val cs = values._1 / (scala.math.sqrt(values._2) * scala.math.sqrt(values._3))
    cs
  }

  /** Generate a vector representation for a sentence
    *
    * @param doc_tfifd_terms a map with the following informations:
    *                        Map(<term_id> -> (<term_raw_freq>, <term_id>, <num_docs_with_terms>, <term_tfidf>))
    *  @param dictionary a broadcast variable with the complete dictionary of terms:
    *                    (term, (term_unique_id, num_docs_with_terms))
    * @return a SparseVector where each element represent a term and the value is the raw frequency weighted by
    *         the TF-IDF
    */
  def generateVector(doc_tfifd_terms: Map[String, Tuple4[Long, Long, Long, Double]],
                             dictionary: Broadcast[Map[String, Tuple2[Long, Long]]]
                            ): SparseVector = {
    val array_values : Seq[(Int, Double)] = doc_tfifd_terms.map(term => {
      val term_id : Int = dictionary.value.getOrElse(term._1, (-1: Long, -1: Long))._1.toInt
      val weight : Double = term._2._4
      (term_id, weight)
    }).filter(_._1 != -1).toSeq
    val v : SparseVector = Vectors.sparse(dictionary.value.size, array_values).toSparse
    v
  }

}
