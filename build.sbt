import NativePackagerHelper._

name := "clustering"

version := "master"

organization := "io.elegans"

scalaVersion := "2.11.8"


resolvers ++= Seq("Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
                  Resolver.bintrayRepo("hseeberger", "maven"))

//https://mvnrepository.com/artifact/org.elasticsearch

libraryDependencies ++= {
  val elastic_client_version = "5.2.2"
  val spark_version = "2.0.2"
  Seq(
    "org.apache.spark" %% "spark-core" % spark_version % "provided",
    "org.apache.spark" %% "spark-mllib" % spark_version % "provided",
    "org.elasticsearch" % "elasticsearch-spark-20_2.11" % elastic_client_version,
    "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0",
    "edu.stanford.nlp" % "stanford-corenlp" % "3.7.0" classifier "models",
    "com.github.scopt" %% "scopt" % "3.5.0",
    "info.bliki.wiki" % "bliki-core" % "3.1.0"
  )
}

SparkSubmit.settings

enablePlugins(JavaServerAppPackaging)

// Assembly settings
mainClass in Compile := Some("io.elegans.clustering.CalcLDA")
mainClass in assembly := Some("io.elegans.clustering.CalcLDA")

mappings in Universal ++= {
  // copy configuration files to config directory
  directory("scripts")
}

assemblyMergeStrategy in assembly := {
	case PathList("META-INF", xs @ _*) => MergeStrategy.discard
	case x => MergeStrategy.first
}

licenses := Seq(("GPLv2", url("https://www.gnu.org/licenses/old-licenses/gpl-2.0.md")))

