import sbtsparksubmit.SparkSubmitPlugin.autoImport._

object SparkSubmit {
  lazy val settings = SparkSubmitSetting(
    SparkSubmitSetting("clustering",
      Seq("--class", "io.elegans.clustering.EsSparkApp"))
  )
}
