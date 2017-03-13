package io.elegans.clustering

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.broadcast.Broadcast

/* import core nlp */
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._

/* these are necessary since core nlp is a java library */
import java.util.Properties
import scala.collection.JavaConversions._

class TextProcessingUtils {

  /** Instantiate a StanfordCoreNLP pipeline
    *
    * @return an instance of StanfordCoreNLP
    */
  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.setProperty("annotators", "tokenize, ssplit")
    val pipeline: StanfordCoreNLP = new StanfordCoreNLP(props)
    pipeline
  }

  /** check if a string is only made by letters
    *
    * @param str the input string
    * @return true if the string is only letters, otherwise false
    */
  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }

  /** tokenize a string, optionally remove stopwords using StanfordCoreNLP
    *
    * @param text the input text
    * @param stopWords the set of stopwords
    * @param pipeline the instance of StanfordCoreNLP
    * @param min_token_length the minimum length for the tokens, default is 0
    * @return a List of lowercase token
    */
  def plainTextToLemmas(text: String, stopWords: Broadcast[Set[String]],
                        pipeline: StanfordCoreNLP, min_token_length: Int = 0): List[String] = {
    val doc: Annotation = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences;
         token <- sentence.get(classOf[TokensAnnotation])) {
      val lemma: String = token.get(classOf[TextAnnotation])
      val lc_lemma = lemma.toLowerCase
      if (lc_lemma.length > min_token_length &&  !stopWords.value.contains(lc_lemma) && isOnlyLetters(lc_lemma)) {
        lemmas += lc_lemma
      }
    }
    lemmas.toList
  }

  /** tokenize a sentence
    *
    * @param text the input string
    * @param stopWords the set of stopwords
    * @param min_token_length the minimum length for the tokens, default is 0
    * @return a list of tokens
    */
  def tokenizeSentence(text: String, stopWords: Broadcast[Set[String]], min_token_length: Int = 0): List[String] = {
    try {
      val pipeline = createNLPPipeline() /* get an instance of the NLP pipeline */
      val doc_lemmas : List[String] =  /* call the tokenization function */
        plainTextToLemmas(text, stopWords, pipeline, min_token_length)
      doc_lemmas /* return the original sentence annotated with tokens */
    } catch {
      case e: Exception => List.empty[String]
    }
  }

}
