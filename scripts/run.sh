#!/usr/bin/env bash 

SCRIPT_PATH=$(dirname `readlink -s -e $0`)
SCRIPT_NAME=$(basename `readlink -s -e $0`)
BUILD_PATH=$(readlink -s -e "$SCRIPT_PATH/../" )

SPARK_MEM=${SPARK_MEM:-8192m}
export SPARK_MEM

JAVA_OPTS="$JAVA_OPTS -Xms$SPARK_MEM -Xmx$SPARK_MEM"

if [[ $# -lt 1 ]]; then
  echo "${SCRIPT_NAME} <classname> <arguments>"
  echo "e.g. ${SCRIPT_NAME} io.elegans.clustering.EsSparkApp --help"
  exit 1
fi

CLASS_NAME=${1}
shift

date
spark-submit --class ${CLASS_NAME} --jars ${HOME}/.ivy2/cache/com.github.scopt/scopt_2.11/jars/scopt_2.11-3.5.0.jar,${HOME}/.ivy2/cache/edu.stanford.nlp/stanford-corenlp/jars/stanford-corenlp-3.6.0-models.jar,${HOME}/.ivy2/cache/edu.stanford.nlp/stanford-corenlp/jars/stanford-corenlp-3.6.0.jar,${HOME}/.ivy2/cache/org.elasticsearch/elasticsearch-spark_2.11/jars/elasticsearch-spark_2.11-2.4.0.jar,${HOME}/.ivy2/cache/org.apache.spark/spark-mllib_2.11/jars/spark-mllib_2.11-2.0.0.jar ${BUILD_PATH}/target/scala-2.11/clustering.11-0.1.jar $@
date

