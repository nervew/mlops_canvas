from feature_engineering.pipeline import FeatureEngineeringPipeline


def test_pipeline_runs(tmp_path):
    pipeline = FeatureEngineeringPipeline()
    pipeline.metrics.export = lambda df, path: None  # skip file writing
    pipeline.drift.compute = lambda ref, new: {"dummy": 0.0}
    pipeline.run()
