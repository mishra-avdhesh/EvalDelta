"""Optional MLflow logging of run metadata and aggregates only (never items or outputs)."""

from __future__ import annotations

from typing import Any

from evaldelta.schemas import RunReport


def log_report(report: RunReport, experiment: str = "evaldelta") -> None:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the 'mlflow' extra: pip install 'evaldelta[mlflow]'") from exc
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=report.run_id):
        params: dict[str, Any] = {
            "config_hash": report.config_hash,
            "split_hash": report.split_hash,
            "old_version_id": report.old_version_id,
            "new_version_id": report.new_version_id,
            "protocol_version": report.protocol_version,
        }
        mlflow.log_params({k: str(v) for k, v in params.items()})
        metrics = {
            "paid_calls": float(report.total_paid_calls),
            "exit_code": float(report.exit_code),
        }
        g = report.global_result
        if g is not None:
            for k in ("effect", "lower", "upper"):
                v = getattr(g, k)
                if v is not None:
                    metrics[f"global_{k}"] = float(v)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("decision", report.decision.value)
