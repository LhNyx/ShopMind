"""评测 runner：检索栈评测（无 LLM、确定性）与全链路评测（真 LLM）两种模式。

retrieval 模式用数据集里的 golden requirements 直接驱动检索器——
隔离 LLM 波动，专测「过滤 + 召回 + 排序」栈，CI 可跑。
live 模式走全图，额外度量理解误差，供人工回归分析。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from shopmind.catalog.store import CatalogStore
from shopmind.eval.metrics import ClarifyCase, EvalReport, mrr, recall_at_k
from shopmind.retrieval.retriever import HybridRetriever
from shopmind.schemas import Requirements, compute_missing


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            import json

            rows.append(json.loads(line))
    return rows


def run_retrieval_eval(
    dataset: list[dict],
    retriever: HybridRetriever,
    store: CatalogStore,
    *,
    ks: tuple[int, ...] = (5, 10),
    mode: str = "retrieval",
) -> EvalReport:
    rows: list[dict] = []
    badcases: list[dict] = []
    clarify_cases: list[ClarifyCase] = []
    latencies: list[float] = []

    for i, row in enumerate(dataset):
        req = Requirements.model_validate(row["requirements"])
        expected = set(row.get("expected_ids", []))
        should_clarify = bool(row.get("should_clarify", False))

        t0 = time.perf_counter()
        pairs, trace = retriever.retrieve(req)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        hits = [p.id for p, _ in pairs]
        actual_clarify = bool(compute_missing(req))
        clarify_cases.append(ClarifyCase(row.get("id", f"q{i}"), should_clarify, actual_clarify))

        row_out = {
            "id": row.get("id", f"q{i}"),
            "query": row.get("query", ""),
            "expected": sorted(expected),
            "hits_top5": hits[:5],
            **{f"recall@{k}": recall_at_k(expected, hits, k) for k in ks},
            "mrr": mrr(expected, hits),
            "after_filter": trace.after_filter,
            "relaxations": trace.relaxations,
            "latency_ms": round(latency_ms, 1),
        }
        rows.append(row_out)
        if expected and row_out[f"recall@{ks[0]}"] < 1.0:
            badcases.append(
                {
                    "id": row_out["id"],
                    "query": row_out["query"],
                    "requirements": req.model_dump(exclude_none=True),
                    "expected": sorted(expected),
                    "got_top10": hits[:10],
                    "after_filter": trace.after_filter,
                    "relaxations": trace.relaxations,
                }
            )

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = _mean(r[f"recall@{k}"] for r in rows)
        metrics[f"hit_rate@{k}"] = _mean(
            1.0 if set(r["expected"]) & set(r["hits_top5"][:k]) else 0.0
            for r in rows
            if r["expected"]
        )
    metrics["mrr"] = _mean(r["mrr"] for r in rows)
    metrics["clarify_accuracy"] = _mean(
        float(c.should_clarify == c.actual_clarify) for c in clarify_cases
    )
    metrics["mean_latency_ms"] = round(_mean(latencies), 1)
    metrics["filter_zero_cases"] = sum(1 for r in rows if r["after_filter"] == 0)

    return EvalReport(
        meta={
            "mode": mode,
            "catalog_version": store.version,
            "catalog_count": len(store.products),
            "n_queries": len(rows),
            "ks": list(ks),
            "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        metrics=metrics,
        rows=rows,
        badcases=badcases,
    )


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0
