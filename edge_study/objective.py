"""Optuna objective: score + factory wired to edge_study settings."""

from __future__ import annotations

from typing import Any, Callable

from edge_study import settings
from edge_study.adb_benchmark import edge_benchmark_trial
from edge_study.export_pte import edge_export_model
from edge_study.host_eval import edge_host_val_sanity_check
from edge_study.optuna_search import edge_optuna_config
from edge_study.prune_optimize import edge_optimise_model


def compute_objective_score(acc: float, latency: float, memory: float) -> float:
    """Same formula as the Optuna objective in EDGE_optuna_study.ipynb."""
    return acc - 0.02 * (latency / 30.0) - 0.001 * (memory / 2000.0)


def make_objective(
    *,
    eval_split: str | None = None,
    dataset_path: str | None = None,
) -> Callable[[Any], float]:
    """Return ``objective(trial)`` using current ``edge_study.settings`` (override args optional)."""

    def objective(trial) -> float:
        config = edge_optuna_config(trial)
        model = edge_optimise_model(config, enable_qat=False)
        print("Model is pruned (FP32); XNNPACK PT2E quant runs in edge_export_model")

        host_acc = edge_host_val_sanity_check(model)
        if host_acc is not None:
            print(f"  [host sanity] ~top1 over few train batches: {host_acc:.4f}")
            trial.set_user_attr("host_train_sanity_top1", host_acc)
            if host_acc < 0.01:
                print(
                    "  [host sanity] WARN: very low — check data path, .pt checkpoint, or prune strength."
                )
        else:
            print("  [host sanity] skipped (Tiny ImageNet train folder missing)")

        pte_path = edge_export_model(model, trial.number, config)
        sp = eval_split if eval_split is not None else settings.EDGE_EVAL_SPLIT
        metrics = edge_benchmark_trial(
            trial.number,
            pte_path,
            dataset_path=dataset_path,
            split=sp,
        )

        acc = float(metrics.get("top1_acc", 0.0))
        latency = float(metrics.get("latency_p95_ms", 999.0))
        memory = float(metrics.get("memory_peak_mb", 9999.0))
        n_samp = int(metrics.get("num_samples", 0))
        v_fwd = int(metrics.get("valid_forward_count", 0))
        dec_fail = int(metrics.get("decode_failures", 0))
        m_err = metrics.get("error")
        m_stat = metrics.get("status", "")

        score = compute_objective_score(acc, latency, memory)

        trial.set_user_attr("top1_acc", acc)
        trial.set_user_attr("latency_ms", latency)
        trial.set_user_attr("memory_mb", memory)
        trial.set_user_attr("num_samples", n_samp)
        trial.set_user_attr("valid_forward_count", v_fwd)
        trial.set_user_attr("decode_failures", dec_fail)
        trial.set_user_attr("metrics_status", m_stat)
        if m_err:
            trial.set_user_attr("metrics_error", m_err)
        for k, v in config.items():
            trial.set_user_attr(k, v)

        print(
            f"  → acc={acc:.3f}, lat={latency:.1f}ms, score={score:.3f} | "
            f"samples={n_samp}, forwards={v_fwd}, decode_fail={dec_fail}, status={m_stat!r}"
        )
        if m_err:
            print(f"  → metrics error field: {m_err}")
        return score

    return objective
