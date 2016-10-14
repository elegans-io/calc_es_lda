import NativePackagerHelper._

name := "calc_es_lda"

version := "0.1"

organization := "io.elegans"

scalaVersion := "2.11.8"

resolvers ++= Seq("Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
                  Resolver.bintrayRepo("hseeberger", "maven"))

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "2.0.0" % "provided",
	"org.apache.spark" %% "spark-mllib" % "2.0.0" % "provided",
	"org.elasticsearch" % "elasticsearch-spark_2.11" % "2.4.0",
	"edu.stanford.nlp" % "stanford-corenlp" % "3.6.0",
	"edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" classifier "models",
	"com.github.scopt" %% "scopt" % "3.5.0"
)

SparkSubmit.settings

enablePlugins(JavaServerAppPackaging)

// Assembly settings
mainClass in Compile := Some("io.elegans.calc_es_lda.EsSparkApp")
mainClass in assembly := Some("io.elegans.calc_es_lda.EsSparkApp")

mappings in Universal ++= {
  // copy configuration files to config directory
  directory("scripts")
}

assemblyMergeStrategy in assembly := {
	case PathList("META-INF", xs @ _*) => MergeStrategy.discard
	case x => MergeStrategy.first
}

licenses := Seq(("GPLv3", url("https://opensource.org/licenses/MIT")))

