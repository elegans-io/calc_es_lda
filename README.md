# Calculate LDA with spark mllib from a dataset stored on elasticsearch 

## Build

Note: works with java 7 and 8 (not with jdk 9)

sbt package

## Classes

### io.elegans.clustering.CalcLDA

```bash
calculate LDA with data from an elasticsearch index.
Usage: LDA with ES data [options]

  --hostname <value>       the hostname of the elasticsearch instance  default: localhost
  --port <value>           the port of the elasticsearch instance  default: 9200
  --group_by_field <value>
                           group the search results by field e.g. conversation, None => no grouping  default: None
  --search_path <value>    the search path on elasticsearch e.g. <index name>/<type name>  default: jenny-en-0/question
  --query <value>          a json string with the query  default: { "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }
  --min_k <value>          min number of topics. default: 8
  --max_k <value>          max number of topics. default: 10
  --maxTermsPerTopic <value>
                           the max number of terms per topic. default: 10
  --maxIterations <value>  number of iterations of learning. default: 100
  --stopwordFile <value>   filepath for a list of stopwords. Note: This must fit on a single machine.  default: Some(stopwords/en_stopwords.txt)
  --used_fields <value>    list of fields to use for LDA, if more than one they will be merged  default: List(question, answer)
  --outputDir <value>      the where to store the output files: topics and document per topics  default: /tmp
  --max_topics_per_doc <value>
                           write the first n topic classifications per document  default: 10
```

#### Output

For each topic number (from min_k to max_k) the program will produce:
* a list of topics with terms and weights -> (num_of_topics, topic_id, term_id, term, weight) e.g. (10,0,1862,help,0.12943777743269205)
* the topic classification for all documents -> (num_of_topics, doc_id, topic_id, weight) e.g. (10,841,1,0.21606734702770736)

For instance, if min_k=5 and max_k=10 the program will produce the following directories:
```
TOPICS_K.10_DN.19699_VS.21974  TOPICS_K.7_DN.19699_VS.21974  TOPICSxDOC_K.10_DN.19699_VS.21974  TOPICSxDOC_K.7_DN.19699_VS.21974
TOPICS_K.5_DN.19699_VS.21974   TOPICS_K.8_DN.19699_VS.21974  TOPICSxDOC_K.5_DN.19699_VS.21974   TOPICSxDOC_K.8_DN.19699_VS.21974
TOPICS_K.6_DN.19699_VS.21974   TOPICS_K.9_DN.19699_VS.21974  TOPICSxDOC_K.6_DN.19699_VS.21974   TOPICSxDOC_K.9_DN.19699_VS.21974
```
where:
* TOPICS_K.10_DN.19699_VS.21974 will contains 10 topics calculated with 19699 documents and with a vocabulary size with 21974 entries
* TOPICSxDOC_K.10_DN.19699_VS.21974 will contains the topic classification for all the 19699 documents with vocabulary size with 21974 entries

The program will print out on the standard output the log likelihood and the prior log probability e.g.:
```
K(5) LOGLIKELIHOOD(-4844389.232238058) LOG_PRIOR(-184.60585976284716) AVGLOGLIKELIHOOD(-245.9205661321924) AVGLOG_PRIOR(-0.009371331527633238) NumDocs(19699) VOCABULAR_SIZE(21974)
K(6) LOGLIKELIHOOD(-4833607.107354659) LOG_PRIOR(-207.6231790028576) AVGLOGLIKELIHOOD(-245.3732223643159) AVGLOG_PRIOR(-0.010539782679468887) NumDocs(19699) VOCABULAR_SIZE(21974)
K(7) LOGLIKELIHOOD(-4808667.540709441) LOG_PRIOR(-227.87874643463218) AVGLOGLIKELIHOOD(-244.1071902487152) AVGLOG_PRIOR(-0.011568036267558363) NumDocs(19699) VOCABULAR_SIZE(21974)
K(8) LOGLIKELIHOOD(-4778600.550597267) LOG_PRIOR(-244.18339819000468) AVGLOGLIKELIHOOD(-242.58086961760836) AVGLOG_PRIOR(-0.012395725579471276) NumDocs(19699) VOCABULAR_SIZE(21974)
K(9) LOGLIKELIHOOD(-4765956.969051744) LOG_PRIOR(-260.1458429346249) AVGLOGLIKELIHOOD(-241.9390308671376) AVGLOG_PRIOR(-0.013206043095315747) NumDocs(19699) VOCABULAR_SIZE(21974)
K(10) LOGLIKELIHOOD(-4758092.727191508) LOG_PRIOR(-275.11088412794464) AVGLOGLIKELIHOOD(-241.53981050771654) AVGLOG_PRIOR(-0.013965728419104758) NumDocs(19699) VOCABULAR_SIZE(21974)
```

#### Run spark job

```bash
sbt "sparkSubmit --class io.elegans.clustering.CalcLDA -- --hostname <es hostname> --group_by_field <field for grouping> --search_path <index_name/type_name> --min_k <min_topics> --max_k <max_topics> --stopwordFile </path/of/stopword_list.txt> --outputDir </path/of/existing/empty/directory>"
```

e.g.

```bash
sbt "sparkSubmit --class io.elegans.clustering.CalcLDA -- --hostname elastic-0.getjenny.com --group_by_field conversation --search_path english/question --min_k 10 --max_k 30 --stopwordFile /tmp/english_stopwords.txt --outputDir /tmp/lda_results"
```

### io.elegans.clustering.KMeansW2VClustering

```bash
calculate clusters with data from an elasticsearch index using w2v representation of phrases.
Usage: Clustering with ES data [options]

  --hostname <value>       the hostname of the elasticsearch instance  default: localhost
  --port <value>           the port of the elasticsearch instance  default: 9200
  --group_by_field <value>
                           group the search results by field e.g. conversation, None => no grouping  default: None
  --search_path <value>    the search path on elasticsearch e.g. <index name>/<type name>  default: jenny-en-0/question
  --query <value>          a json string with the query  default: { "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }
  --stopwordFile <value>   filepath for a list of stopwords. Note: This must fit on a single machine.  default: Some(stopwords/en_stopwords.txt)
  --used_fields <value>    list of fields to use for LDA, if more than one they will be merged  default: List(question, answer)
  --outputDir <value>      the where to store the output files: topics and document per topics  default: /tmp
  --min_k <value>          min number of topics. default: 8
  --max_k <value>          max number of topics. default: 10
  --maxIterations <value>  number of iterations of learning. default: 10
  --inputW2VModel <value>  the input word2vec model
```

### io.elegans.clustering.TrainW2V

```bash
Train a W2V model taking input data from ES.
Usage: Train a W2V model [options]

  --hostname <value>       the hostname of the elasticsearch instance  default: localhost
  --port <value>           the port of the elasticsearch instance  default: 9200
  --group_by_field <value>
                           group the search results by field e.g. conversation, None => no grouping  default: None
  --search_path <value>    the search path on elasticsearch e.g. <index name>/<type name>  default: jenny-en-0/question
  --query <value>          a json string with the query  default: { "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }
  --stopwordFile <value>   filepath for a list of stopwords. Note: This must fit on a single machine.  default: Some(stopwords/en_stopwords.txt)
  --used_fields <value>    list of fields to use for LDA, if more than one they will be merged  default: List(question, answer)
  --outputDir <value>      the where to store the output files: topics and document per topics  default: /tmp
  --vector_size <value>    the vector size  default: 300
  --word_window_size <value>
                           the word window size  default: 5
  --learningRate <value>   the learningRate  default: 0.025
```

### io.elegans.clustering.W2VModelToSparkFormat

```bash
Load word2vec model in textual format separated by spaces (<term> <v0> .. <vn>) and save it in spark format.
Usage: W2VModelToSparkFormat [options]

  --inputfile <value>  the file with the model
  --outputdir <value>  the port of the elasticsearch instance
```

### io.elegans.clustering.ReduceW2VModel

```bash
Generate a reduced W2V model by selecting only the vectors of the words used in the dataset
Usage: ReduceW2VModel [options]

  --hostname <value>       the hostname of the elasticsearch instance  default: localhost
  --port <value>           the port of the elasticsearch instance  default: 9200
  --group_by_field <value>
                           group the search results by field e.g. conversation, None => no grouping  default: None
  --search_path <value>    the search path on elasticsearch e.g. <index name>/<type name>  default: jenny-en-0/question
  --query <value>          a json string with the query  default: { "fields":["question", "answer", "conversation", "index_in_conversation", "_id" ] }
  --stopwordFile <value>   filepath for a list of stopwords. Note: This must fit on a single machine.  default: Some(stopwords/en_stopwords.txt)
  --used_fields <value>    list of fields to use for LDA, if more than one they will be merged  default: List(question, answer)
  --inputfile <value>      the file with the model
  --outputfile <value>     the output file  default: /tmp/w2v_model.txt
```