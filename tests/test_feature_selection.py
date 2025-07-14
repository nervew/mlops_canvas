from feature_selection_pipeline import run


def test_pipeline_runs():
    df = run()
    assert not df.empty
    assert df.shape[1] <= 4
