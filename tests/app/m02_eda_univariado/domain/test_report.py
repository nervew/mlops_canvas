from app.m02_eda_univariado.domain.report import UnivariateReport


def test_univariate_report_holds_description():
    report = UnivariateReport({"a": 1})
    assert report.description["a"] == 1
