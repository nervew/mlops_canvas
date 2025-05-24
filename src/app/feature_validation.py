import great_expectations as ge
import json
import os

class FeatureValidatorBase:
    def __init__(self, expectation_suite_name: str, expectations_path: str = "/dbfs/great_expectations/"):
        self.expectation_suite_name = expectation_suite_name
        self.expectations_path = expectations_path
        self.context = ge.data_context.DataContext(root_directory=self.expectations_path)
    
    def save_expectation_suite(self, suite):
        self.context.save_expectation_suite(suite, self.expectation_suite_name + ".json")

    def load_expectation_suite(self):
        return self.context.get_expectation_suite(self.expectation_suite_name)
    
    def save_report(self, report: dict, filename: str):
        report_path = os.path.join(self.expectations_path, filename)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Reporte guardado en: {report_path}")

    def load_report(self, filename: str):
        report_path = os.path.join(self.expectations_path, filename)
        with open(report_path) as f:
            return json.load(f)

class FeatureValidatorTrain(FeatureValidatorBase):
    def __init__(self, expectation_suite_name: str, expectations_path: str = "/dbfs/great_expectations/"):
        super().__init__(expectation_suite_name, expectations_path)

    def fit(self, df):
        """
        Crea expectativas automáticas a partir del DataFrame y guarda expectation suite y reporte.
        """
        # Convertir Pandas DataFrame a GE Dataset
        ge_df = ge.from_pandas(df)
        # Generar expectation suite automática (por ejemplo, usando profiling)
        suite = ge_df.get_expectation_suite(discard_failed_expectations=False, expectation_suite_name=self.expectation_suite_name)

        # Guardar expectation suite en disco
        self.save_expectation_suite(suite)

        # Validar el mismo dataset para generar reporte inicial
        results = ge_df.validate(expectation_suite=suite)
        
        # Guardar reporte JSON con resultados
        report = {
            "success": results.success,
            "statistics": results.statistics,
            "results": results.results
        }
        self.save_report(report, "initial_validation_report.json")
        return report

class FeatureValidatorCheck(FeatureValidatorBase):
    def __init__(self, expectation_suite_name: str, expectations_path: str = "/dbfs/great_expectations/"):
        super().__init__(expectation_suite_name, expectations_path)

    def validate(self, df):
        """
        Carga expectation suite guardada y valida el dataframe entrante,
        genera reporte JSON y devuelve booleano de correcto o no.
        """
        # Convertir Pandas DataFrame a GE Dataset
        ge_df = ge.from_pandas(df)

        # Cargar expectation suite guardada
        suite = self.load_expectation_suite()

        # Validar dataset
        results = ge_df.validate(expectation_suite=suite)

        # Guardar reporte JSON
        report = {
            "success": results.success,
            "statistics": results.statistics,
            "results": results.results
        }
        self.save_report(report, "pipeline_validation_report.json")
        return results.success, report
