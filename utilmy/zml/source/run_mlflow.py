# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-

import mlflow
from source.util_feature import load, log


def register(run_name, params, metrics, signature, model_class, tracking_uri= "sqlite:///local.db"):
    """
    :run_name: Name of model
    :log_params: dict with model params
    :metrics: dict with model evaluation metrics
    :signature: Its a signature that describes model input and output Schema
    :model_class: Type of class model
    :return:
    """
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name=run_name) as run:
        run_id        = run.info.run_uuid
        experiment_id = run.info.experiment_id

        sk_model      = load(params['path_train_model'] + "/model.pkl")
        mlflow.log_params(params)

        metrics.apply(lambda x: mlflow.log_metric(x.metric_name, x.metric_val), axis=1)

        mlflow.sklearn.log_model(sk_model, run_name, signature=signature,
                                 registered_model_name="sklearn_"+run_name+"_"+model_class)

        log("MLFLOW identifiers", run_id, experiment_id)

    mlflow.end_run()
