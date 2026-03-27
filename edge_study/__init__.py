"""EDGE Optuna + ExecuTorch pipeline (extracted from EDGE_optuna_study.ipynb)."""

from edge_study.adb_benchmark import (
    clear_edge_metrics_json_on_device,
    edge_benchmark_trial,
    edge_poll_metrics,
    move_dataset_to_app,
    move_pte_to_app,
)
from edge_study.export_pte import edge_export_model
from edge_study.metrics_json import _adb_cat_stdout_not_metrics_json, _metrics_payload_ready
from edge_study.model_loaders import (
    _load_mobilenetv3_from_pt,
    _load_resnet18_from_pt,
    _new_module_from_cpu_state_dict,
    load_model_and_train,
)
from edge_study.objective import compute_objective_score, make_objective
from edge_study.optuna_search import edge_optuna_config
from edge_study.prune_optimize import apply_torch_l1_prune, edge_optimise_model
from edge_study.quant_xnnpack import (
    LAYER_QUANT_SCHEME_EXPORT_MAP,
    QUANT_LAYER_NAMES,
    XNNPACK_LAYER_QUANT_CHOICES,
    list_xnnpack_quantizable_module_names,
    optuna_param_name_for_module,
    resolve_layer_quant_scheme_for_export,
    scheme_to_xnnpack_config,
)
from edge_study.host_eval import edge_host_val_sanity_check, tiny_train_root

__all__ = [
    "QUANT_LAYER_NAMES",
    "XNNPACK_LAYER_QUANT_CHOICES",
    "LAYER_QUANT_SCHEME_EXPORT_MAP",
    "apply_torch_l1_prune",
    "clear_edge_metrics_json_on_device",
    "compute_objective_score",
    "edge_benchmark_trial",
    "edge_export_model",
    "edge_host_val_sanity_check",
    "edge_optimise_model",
    "edge_optuna_config",
    "edge_poll_metrics",
    "list_xnnpack_quantizable_module_names",
    "load_model_and_train",
    "make_objective",
    "move_dataset_to_app",
    "move_pte_to_app",
    "optuna_param_name_for_module",
    "resolve_layer_quant_scheme_for_export",
    "scheme_to_xnnpack_config",
    "tiny_train_root",
    "_adb_cat_stdout_not_metrics_json",
    "_load_mobilenetv3_from_pt",
    "_load_resnet18_from_pt",
    "_metrics_payload_ready",
    "_new_module_from_cpu_state_dict",
]
