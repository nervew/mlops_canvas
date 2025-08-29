import matplotlib.pyplot as plt
import mlflow
import pytest
from unittest import mock

from mlops.mlflow_helper import MLflowHelper


def _create_helper():
    helper = MLflowHelper("exp", default_tags={"project": "demo"})
    helper.experiment_id = "1"
    return helper


def test_configure_creates_experiment(mocker):
    helper = MLflowHelper("exp")
    set_tracking = mocker.patch("mlflow.set_tracking_uri")
    set_registry = mocker.patch("mlflow.set_registry_uri")
    get_exp = mocker.patch("mlflow.get_experiment_by_name", return_value=None)
    create_exp = mocker.patch("mlflow.create_experiment", return_value="1")
    helper.configure()
    set_tracking.assert_called_once_with(helper.tracking_uri)
    set_registry.assert_called_once_with(helper.registry_uri)
    get_exp.assert_called_once_with("exp")
    create_exp.assert_called_once_with("exp")
    assert helper.experiment_id == "1"


def test_run_context_manager_calls_start_and_end(mocker):
    helper = _create_helper()
    start = mocker.patch("mlflow.start_run")
    end = mocker.patch("mlflow.end_run")
    with helper.run("run1", tags={"user": "alice"}):
        pass
    start.assert_called_once()
    end.assert_called_once()
    called_tags = start.call_args.kwargs["tags"]
    assert called_tags["project"] == "demo"
    assert called_tags["user"] == "alice"


def test_nested_runs(mocker):
    helper = _create_helper()
    start = mocker.patch("mlflow.start_run")
    helper.start_run("parent")
    helper.start_run("child", nested=True)
    assert start.call_args_list[0].kwargs["nested"] is False
    assert start.call_args_list[1].kwargs["nested"] is True


def test_logging_methods(mocker, tmp_path):
    helper = _create_helper()
    lp = mocker.patch("mlflow.log_params")
    lm = mocker.patch("mlflow.log_metrics")
    la = mocker.patch("mlflow.log_artifacts")
    lf = mocker.patch("mlflow.log_figure")
    st = mocker.patch("mlflow.set_tags")

    helper.log_params({"lr": 0.1})
    helper.log_metrics({"mae": 0.5}, step=1)
    helper.log_artifacts(str(tmp_path))
    fig = plt.figure()
    helper.log_figure(fig, "plot.png")
    helper.set_tags({"stage": "train"})

    lp.assert_called_once()
    lm.assert_called_once_with({"mae": 0.5}, step=1)
    la.assert_called_once()
    lf.assert_called_once()
    st.assert_called_once_with({"stage": "train"})


def test_enable_autolog(mocker):
    helper = _create_helper()
    sk = mocker.patch("mlflow.sklearn.autolog")
    helper.enable_autolog(["sklearn", "xgboost"])
    sk.assert_called_once()


def test_register_and_log_model(mocker):
    helper = _create_helper()
    register = mocker.patch("mlflow.register_model")
    log_model = mocker.patch("mlflow.pyfunc.log_model")
    active_run = mocker.patch(
        "mlflow.active_run",
        return_value=mock.Mock(info=mock.Mock(run_id="r1")),
    )
    helper.log_and_register_model(object(), "Model")
    log_model.assert_called_once()
    register.assert_called_once_with("runs:/r1/model", "Model")
    active_run.assert_called()


def test_retry_logic(mocker):
    helper = _create_helper()
    call = mocker.patch(
        "mlflow.log_params", side_effect=[Exception("boom"), None]
    )
    sleep = mocker.patch("time.sleep")
    helper.log_params({"a": 1})
    assert call.call_count == 2
    sleep.assert_called_once()
