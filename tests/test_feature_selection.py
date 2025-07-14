from feature_selection.run import run_pipeline


def test_pipeline_runs():
    df = run_pipeline()
    assert not df.empty
    assert 0 < df.shape[1] <= 4
