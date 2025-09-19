# m08_feature_reduction/pipeline_reduction.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnxruntime as ort
from onnx.defs import onnx_opset_version
from onnx import version_converter

from .step01_ingesta.adapters.parquet_loader import ParquetPartitionLoader
from .step06_evaluation.adapters.evaluator import Evaluator


# ───────── localización de particiones ─────────
def find_partition_dir(start: Path) -> Path:
    d = start.resolve()
    while d != d.parent:
        cand = d / "data" / "raw" / "partitioned"
        if (cand / "train_df.parquet").exists():
            return cand
        d = d.parent
    raise FileNotFoundError("No se encontró data/raw/partitioned/*_df.parquet")


THIS_FILE = Path(__file__).resolve()
ENV_ROOT = os.environ.get("MLOPS_ROOT", "").strip()
if ENV_ROOT:
    PROJECT_ROOT = Path(ENV_ROOT).resolve()
    PART_DIR = PROJECT_ROOT / "data" / "raw" / "partitioned"
    if not (PART_DIR / "train_df.parquet").exists():
        raise FileNotFoundError(f"MLOPS_ROOT={PROJECT_ROOT} pero falta data/raw/partitioned.")
else:
    PART_DIR = find_partition_dir(THIS_FILE)
    PROJECT_ROOT = PART_DIR.parent.parent.parent  # …/mlops_canvas

print(f"Buscando particiones en: {PART_DIR}")


# ───────── utilidades ─────────
def _resolve_rel_to_root(p: str | Path | None, default_rel: Path) -> Path:
    if p is None:
        return (PROJECT_ROOT / default_rel).resolve()
    pth = Path(p)
    return pth if pth.is_absolute() else (PROJECT_ROOT / pth).resolve()


def _rglob_one_of(names: List[str]) -> Optional[Path]:
    cands: List[Path] = []
    for nm in names:
        cands.extend(PROJECT_ROOT.rglob(nm))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)


def _read_json_list(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        arr = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    return None


def _load_feature_names_out(path: Path) -> Optional[List[str]]:
    return _read_json_list(path)


def _to_indices_from_any(
    selected: Optional[List[str]],
    k_base: int,
    feature_names_out: Optional[List[str]] = None,
) -> Optional[List[int]]:
    if not selected:
        return None
    idx: List[int] = []
    if feature_names_out:
        pos = {str(n): i for i, n in enumerate(feature_names_out)}
        for s in selected:
            if s in pos:
                idx.append(pos[s])
            else:
                try:
                    j = int(s)
                    if 0 <= j < k_base:
                        idx.append(j)
                except Exception:
                    pass
    else:
        for s in selected:
            try:
                j = int(s)
                if 0 <= j < k_base:
                    idx.append(j)
            except Exception:
                pass
    idx = sorted({i for i in idx if 0 <= i < k_base})
    return idx if idx else None


def _detect_temporal_candidates(
    df: pd.DataFrame, min_valid_ratio: float = 0.8, min_unique: int = 3
) -> list[str]:
    temporal: list[str] = []
    for c in df.columns:
        s = df[c]
        if getattr(s.dtype, "kind", None) == "M":
            temporal.append(c)
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            try:
                parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() >= min_valid_ratio and parsed.nunique(dropna=True) >= min_unique:
                    temporal.append(c)
            except Exception:
                pass
    return list(dict.fromkeys(temporal))


# ───────── ONNX helpers ─────────
def _build_ort_inputs(session: ort.InferenceSession, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    need = [i.name for i in session.get_inputs()]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas por el ONNX: {missing}")
    feed: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        col = df[name]
        t = inp.type.lower()
        if "tensor(float" in t:
            arr = col.to_numpy(dtype=np.float32).reshape(-1, 1)
        elif "tensor(double" in t:
            arr = col.to_numpy(dtype=np.float64).reshape(-1, 1)
        elif "tensor(string" in t:
            arr = col.astype(object).values.reshape(-1, 1)
        else:
            arr = col.to_numpy(dtype=np.float32).reshape(-1, 1)
        feed[name] = arr
    return feed


def _run_onnx_transform(onnx_path: Path, X_df: pd.DataFrame) -> np.ndarray:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.get_outputs()
    if len(outs) != 1:
        raise ValueError(f"El transformador ONNX debe tener 1 salida; tiene {len(outs)}.")
    feed = _build_ort_inputs(sess, X_df)
    out = sess.run(None, feed)[0]
    m = np.asarray(out)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    return m.astype(np.float32)


def _collect_opsets(model: onnx.ModelProto) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for oi in model.opset_import:
        dom = oi.domain or ""
        d[dom] = max(d.get(dom, 0), int(oi.version))
    return d


def _harmonize_for_merge(models: List[onnx.ModelProto]) -> Tuple[List[onnx.ModelProto], Dict[str, int], int]:
    if not models:
        raise ValueError("No hay modelos para armonizar.")
    max_ir = max(int(m.ir_version or 0) for m in models)
    target_ir = min(max_ir, int(onnx.IR_VERSION))

    union: Dict[str, int] = {}
    for m in models:
        for dom, ver in _collect_opsets(m).items():
            union[dom] = max(union.get(dom, 0), int(ver))

    default_ver = max(union.get("", 0), union.get("ai.onnx", 0), 0)
    if default_ver == 0:
        default_ver = min(onnx_opset_version(), 17)
    default_ver = min(default_ver, onnx_opset_version())

    target_opsets: Dict[str, int] = {}
    for dom, ver in union.items():
        if dom in ("", "ai.onnx"):
            target_opsets[""] = default_ver
        else:
            target_opsets[dom] = ver

    adjusted: List[onnx.ModelProto] = []
    for m in models:
        cur = m
        cur_default = max(_collect_opsets(cur).get("", 0), _collect_opsets(cur).get("ai.onnx", 0), 0)
        try:
            if cur_default and cur_default != default_ver:
                cur = version_converter.convert_version(cur, default_ver)
        except Exception:
            pass

        cur.ir_version = target_ir
        new_imports = [oh.make_operatorsetid("", target_opsets[""])]
        for dom, ver in sorted(target_opsets.items()):
            if dom == "":
                continue
            new_imports.append(oh.make_operatorsetid(dom, ver))
        del cur.opset_import[:]
        cur.opset_import.extend(new_imports)
        onnx.checker.check_model(cur)
        adjusted.append(cur)

    return adjusted, target_opsets, target_ir


def _add_prefix_all(
    model: onnx.ModelProto,
    prefix: str,
    keep_inputs: bool = False,
    keep_outputs: bool = False,
) -> onnx.ModelProto:
    return onnx.compose.add_prefix(
        model,
        prefix=prefix,
        rename_nodes=True,
        rename_edges=True,
        rename_inputs=not keep_inputs,
        rename_outputs=not keep_outputs,
        rename_initializers=True,
        rename_value_infos=True,
        rename_functions=True,
        inplace=False,
    )


def _compose_two(m1: onnx.ModelProto, m2: onnx.ModelProto) -> onnx.ModelProto:
    if len(m1.graph.output) != 1 or len(m2.graph.input) != 1:
        raise ValueError("Cada submodelo debe tener una sola salida/entrada.")
    merged = onnx.compose.merge_models(
        m1, m2, io_map=[(m1.graph.output[0].name, m2.graph.input[0].name)]
    )
    onnx.checker.check_model(merged)
    return merged


def _compose_chain_prefixed(models: List[onnx.ModelProto]) -> onnx.ModelProto:
    if not models:
        raise ValueError("No hay modelos para componer.")
    adjusted, _, _ = _harmonize_for_merge(models)
    prefixed: List[onnx.ModelProto] = []
    for i, m in enumerate(adjusted):
        if i == 0:
            prefixed.append(m)
        else:
            prefixed.append(_add_prefix_all(m, prefix=f"m{i}_", keep_inputs=False, keep_outputs=False))
    merged = prefixed[0]
    for nxt in prefixed[1:]:
        merged = _compose_two(merged, nxt)
    onnx.checker.check_model(merged)
    return merged


# ───────── orquestación m08 (sin insertar selector) ─────────
def run_feature_reduction(
    transformer_inicial_onnx_path: str | Path | None = None,
    modelo_onnx_path: str | Path | None = None,
    feature_names_out_path: str | Path | None = None,
    lista_features_global_path: str | Path | None = None,  # lista_engineering.json
    lista_features_m08_path: str | Path | None = None,  # salida m08 (reduction)
    target_var: str = "target",
    verbose: bool = True,
    # compat con versiones previas (ignorados si vienen)
    transformer: Any = None,
    features: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    use_reduction: Optional[bool] = None,
    temporal_vars: Optional[List[str]] = None,
    mae_anterior: Any = None,
) -> Dict[str, Any]:

    # Defaults
    t_default = Path("transformers/transformador_inicial.onnx")
    m_default = Path("models/modelo_v1.onnx")
    n_default = Path("logs/feature_names_out.json")
    g_default = Path("output/Lista_feature_final/lista_engineering.json")
    r_default = Path("output/Lista_feature_final/lista_reduction.json")

    t_path = _resolve_rel_to_root(transformer_inicial_onnx_path, t_default)
    m_path = _resolve_rel_to_root(modelo_onnx_path, m_default)
    n_path = _resolve_rel_to_root(feature_names_out_path, n_default)
    g_path = _resolve_rel_to_root(lista_features_global_path, g_default)
    r_path = _resolve_rel_to_root(lista_features_m08_path, r_default)

    if not t_path.exists():
        alt = _rglob_one_of(["transformador_inicial.onnx"])
        if alt is None:
            raise FileNotFoundError(f"No existe transformador inicial ONNX en {t_path}")
        t_path = alt
    if not m_path.exists():
        alt = _rglob_one_of(["modelo_v1.onnx", "modelo.onnx"])
        if alt is None:
            raise FileNotFoundError(f"No existe modelo ONNX en {m_path}")
        m_path = alt

    # Ingesta (usa particiones ya creadas por el m04)
    loader = ParquetPartitionLoader(str(PART_DIR))
    X_tr_raw, X_te_raw, _ = loader.load()

    # Separar target si existe
    def drop_target(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        if target_var in df.columns:
            y = df[target_var].copy()
            X = df.drop(columns=[target_var])
            return X, y
        return df.copy(), None

    X_tr_core, y_tr = drop_target(X_tr_raw)
    X_te_core, y_te = drop_target(X_te_raw)

    # Diagnóstico + columnas temporales candidatas
    temporal_cands = _detect_temporal_candidates(X_tr_raw)

    if verbose:
        print("Rutas elegidas:")
        print(f"  • transformador_inicial: {t_path}")
        print(f"  • modelo               : {m_path}")
        print(f"  • feature_names_out    : {n_path if n_path.exists() else '— (no existe)'}")
        print(f"  • lista_engineering    : {g_path if g_path.exists() else '— (no existe)'}")
        print(f"  • lista_reduction(out) : {r_path}")
        print(f"  • particiones          : {PART_DIR}")
        print(f"Shapes crudos: train={X_tr_core.shape}, test={X_te_core.shape}")
        if temporal_cands:
            print(f"Temporales candidatas (auto): {temporal_cands}")

    # Pasar por transformador para conocer k_base (nº de columnas que el modelo espera)
    Z_tr = _run_onnx_transform(t_path, X_tr_core)
    k_base = Z_tr.shape[1]
    if verbose:
        print(f"Transformador inicial → Z_tr: shape={Z_tr.shape} (k_base={k_base})")

    # Leer listas
    feat_names_out = _load_feature_names_out(n_path)  # puede ser None
    sel_engineering = _read_json_list(g_path)         # lista por nombres
    sel_m08_prev = _read_json_list(r_path)            # si existía de corridas anteriores
    sel_legacy = list(features) if features else None

    idx_engineering = _to_indices_from_any(sel_engineering, k_base, feat_names_out)
    idx_prev = _to_indices_from_any(sel_m08_prev, k_base, feat_names_out)
    idx_legacy = _to_indices_from_any(sel_legacy, k_base, feat_names_out)

    # intersección sensata
    lists = [x for x in [idx_engineering, idx_prev, idx_legacy] if x]
    if len(lists) >= 2:
        final_idx = sorted(set(lists[0]).intersection(*lists[1:]))
    elif len(lists) == 1:
        final_idx = lists[0]
    else:
        final_idx = None  # todas

    if verbose:
        if sel_engineering is not None:
            heads = sel_engineering[:10]
            print(
                f"Lista engineering (n={len(sel_engineering)}): "
                f"{heads}{' ...' if len(sel_engineering) > 10 else ''}"
            )
        if final_idx is None:
            print("Selección final: sin listas válidas → TODAS las columnas del transformador.")
        else:
            print(f"Selección final: {len(final_idx)} columnas (de {k_base}). Ejemplo idx: {final_idx[:10]}")

    # Componer ONNX final: SOLO transformador + modelo (sin rebanar columnas)
    tf_inicial = onnx.load(str(t_path))
    modelo = onnx.load(str(m_path))
    prelim_adj, _, _ = _harmonize_for_merge([tf_inicial, modelo])
    final_onnx = _compose_chain_prefixed([prelim_adj[0], prelim_adj[1]])

    # Guardar ONNX end-to-end
    transformers_dir = PROJECT_ROOT / "transformers"
    transformers_dir.mkdir(parents=True, exist_ok=True)
    final_path = transformers_dir / "transformador_final.onnx"
    onnx.save(final_onnx, str(final_path))
    if verbose:
        print(f"✔ ONNX end-to-end guardado en {final_path}")

    # Exportar lista_reduction.json (solo como insumo documental/analítico)
    lista_dir = PROJECT_ROOT / "output" / "Lista_feature_final"
    lista_dir.mkdir(parents=True, exist_ok=True)

    if final_idx is None:
        # todas
        if feat_names_out:
            reduction_names = list(map(str, feat_names_out))
        else:
            reduction_names = [str(i) for i in range(k_base)]
    else:
        if feat_names_out:
            reduction_names = [str(feat_names_out[i]) for i in final_idx]
        else:
            reduction_names = [str(i) for i in final_idx]

    (lista_dir / "lista_reduction.json").write_text(
        json.dumps(reduction_names, indent=4), encoding="utf-8"
    )
    if verbose:
        print(f"Lista reduction exportada: {lista_dir / 'lista_reduction.json'} (n={len(reduction_names)})")
        print(
            f"Primeros nombres reduction: "
            f"{reduction_names[:10]}{' ...' if len(reduction_names) > 10 else ''}"
        )

    # Evaluación rápida en test (si hay target) usando el ONNX end-to-end (sin reducción)
    def _predict_fn_end2end(X_df: pd.DataFrame) -> np.ndarray:
        sess = ort.InferenceSession(str(final_path), providers=["CPUExecutionProvider"])
        feed = _build_ort_inputs(sess, X_df)
        y = sess.run(None, feed)[0]
        return np.asarray(y).ravel()

    evaluator = Evaluator()
    metrics = evaluator.evaluate(_predict_fn_end2end, X_te_core, y_te) if y_te is not None else {}
    if verbose and metrics:
        print(f"Metrics test: {metrics}")

    return {
        "onnx_path": str(final_path),
        "metrics": metrics,
        "k_base": int(k_base),
        "n_selected": int(len(reduction_names)),
        "feature_names_out_used": bool(feat_names_out is not None),
        "lista_engineering_path": str(g_path) if g_path.exists() else None,
        "lista_reduction_path": str(lista_dir / "lista_reduction.json"),
        "temporal_candidates": temporal_cands,
        "reduction_names_head": reduction_names[:20],
    }
