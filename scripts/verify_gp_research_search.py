"""Small real-ledger smoke test of optional GP research search (2024 only)."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Genetic_Algorithm.config import STAGES, load_config
from Genetic_Algorithm.data import load_stage
from Genetic_Algorithm.evaluator import evaluate_tree
from Genetic_Algorithm.evolution import search
from Genetic_Algorithm.expression import Node
from Genetic_Algorithm.features import RAW_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Fixed known research structures, not picked using this smoke's results.
    trees = tuple(Node("rolling_hit_rate", (
        Node("rolling_autocorr", (Node("rolling_min", (Node("body_relative"),), window=minimum),), window=20),
    ), window=outer) for minimum, outer in ((10, 48), (10, 60), (10, 72), (8, 60)))
    config = replace(
        load_config(ROOT / "Genetic_Algorithm/configs/daily_5groups_behavior_elites.json"),
        population=4, generations=1, initial_trees=trees, max_attempts=4,
        training_parameter_stability_top_k=2,
    )
    data = load_stage(args.h5, STAGES["train"], config.max_history, sorted(RAW_FIELDS), include_accounting=True)
    result = search(data, config)
    assert result.evaluations == 4
    assert any(d.get("behavior") for d in result.fitness_diagnostics.values())
    assert any(d.get("return_moments") for d in result.fitness_diagnostics.values())
    assert result.generation_log[-1]["cumulative_parameter_stability_candidates"] > 0
    assert all("training_parameter_stability" in result.fitness_diagnostics[c.expression_id] for c in result.candidates)
    cutoff = "2024-09-01"
    differences = []
    for tree in trees:
        full = evaluate_tree(tree, data.features, data.eligible).loc[:cutoff]
        short = evaluate_tree(tree, {k: v.loc[:cutoff] for k, v in data.features.items()}, data.eligible.loc[:cutoff])
        assert full.isna().equals(short.isna())
        difference = float((full - short).abs().max().max())
        assert difference <= 1e-10
        differences.append(difference)
    report = {
        "status": "passed", "stage": "2024 only", "evaluations": result.evaluations,
        "retained_elites": len(result.candidates), "progress": result.generation_log[-1],
        "cutoff": cutoff, "max_abs_diffs": differences,
        "research": result.research_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "research"}))


if __name__ == "__main__":
    main()
