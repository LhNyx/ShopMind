"""eval 包：端到端评测框架。"""

from shopmind.eval.metrics import EvalReport, mrr, recall_at_k
from shopmind.eval.runner import load_dataset, run_retrieval_eval

__all__ = ["EvalReport", "load_dataset", "mrr", "recall_at_k", "run_retrieval_eval"]
