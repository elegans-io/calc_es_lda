import sbtsparksubmit.SparkSubmitPlugin.autoImport._

object SparkSubmit {
  lazy val settings = SparkSubmitSetting(
    SparkSubmitSetting("calc_es_lda",
      Seq("--class", "io.elegans.calc_es_lda.EsSparkApp"))
  )
}
