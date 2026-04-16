from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ml_schema import (
    candidate_record_from_row,
    config_hash,
    ensure_parent_dir,
    portfolio_decision_record,
    safe_value,
)


class JSONLWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_parent_dir(self.path)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_value(record), ensure_ascii=False) + "\n")

    def append_many(self, records: Iterable[Dict[str, Any]]) -> int:
        count = 0
        with self.path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(safe_value(record), ensure_ascii=False) + "\n")
                count += 1
        return count


class MLDataWriter:
    def __init__(
        self,
        base_dir: str | Path = "./logs/ml",
        *,
        engine_name: str = "oil_engine_daily",
        engine_version: str = "paper_v1",
    ):
        self.base_dir = Path(base_dir)
        self.engine_name = engine_name
        self.engine_version = engine_version

        self.candidate_writer = JSONLWriter(self.base_dir / "ml_candidate_dataset.jsonl")
        self.portfolio_writer = JSONLWriter(self.base_dir / "portfolio_decisions.jsonl")
        self.trade_outcome_writer = JSONLWriter(self.base_dir / "ml_trade_outcomes.jsonl")

    def write_candidate_snapshot(
        self,
        *,
        ranked_df: pd.DataFrame,
        run_id: str,
        cycle_timestamp_et: str,
        config: Dict[str, Any],
        portfolio_plan: Optional[Dict[str, Any]] = None,
        executed_tickers: Optional[set[str]] = None,
    ) -> int:
        if ranked_df is None or ranked_df.empty:
            return 0

        executed_tickers = executed_tickers or set()
        cfg_hash = config_hash(config)

        action_lookup: dict[str, tuple[Optional[str], Optional[str]]] = {}
        if portfolio_plan:
            reason = portfolio_plan.get("reason")
            for action in portfolio_plan.get("actions", []):
                ticker = str(action.get("ticker") or "").strip()
                if ticker:
                    action_lookup[ticker] = (action.get("action"), reason)

        records: List[Dict[str, Any]] = []
        for idx, (_, row) in enumerate(ranked_df.iterrows()):
            ticker = str(row.get("contract_ticker") or "").strip()
            portfolio_action, portfolio_reason = action_lookup.get(ticker, (None, portfolio_plan.get("reason") if portfolio_plan else None))
            records.append(
                candidate_record_from_row(
                    row=row,
                    engine_name=self.engine_name,
                    run_id=run_id,
                    cycle_timestamp_et=cycle_timestamp_et,
                    config_hash_value=cfg_hash,
                    portfolio_action=portfolio_action,
                    portfolio_reason=portfolio_reason,
                    was_top_ranked=idx == 0,
                    was_executed=ticker in executed_tickers,
                    engine_version=self.engine_version,
                )
            )

        written = self.candidate_writer.append_many(records)
        logging.info("ML candidate snapshot written | rows=%s | path=%s", written, self.candidate_writer.path)
        return written

    def write_portfolio_decision(
        self,
        *,
        plan: Dict[str, Any],
        run_id: str,
        cycle_timestamp_et: str,
        config: Dict[str, Any],
    ) -> None:
        cfg_hash = config_hash(config)
        record = portfolio_decision_record(
            plan=plan,
            engine_name=self.engine_name,
            run_id=run_id,
            cycle_timestamp_et=cycle_timestamp_et,
            config_hash_value=cfg_hash,
            engine_version=self.engine_version,
        )
        self.portfolio_writer.append(record)
        logging.info("ML portfolio decision written | recommendation=%s | path=%s", plan.get("recommendation"), self.portfolio_writer.path)

    def write_trade_outcome(self, record: Dict[str, Any]) -> None:
        self.trade_outcome_writer.append(record)
        logging.info("ML trade outcome written | trade_id=%s | path=%s", record.get("trade_id"), self.trade_outcome_writer.path)
