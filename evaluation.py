"""
Evaluation and reporting.

Reports (as required by Roadmap item 3):
    - Accuracy per step
    - Average number of LLM calls
    - False-certainty cases (high agreement but wrong answer)
    - Uncertainty-drops-but-still-wrong cases
"""

import json
import os
from collections import defaultdict

import pandas as pd
import numpy as np

from utils import answers_match

def evaluate_results(results: list[dict], output_dir: str) -> dict:
    """
    Compute all required metrics and save reports.

    Args:
        results: List of result dicts from run_audit_loop, each augmented
                 with 'gold_answer'.
        output_dir: Directory to save reports.

    Returns:
        Summary dict with all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Accuracy per step ─────────────────────────────────────
    step_correct = defaultdict(list)
    step_answers = defaultdict(list)

    for r in results:
        gold = r["gold_answer"]
        for step_data in r["steps"]:
            step_idx = step_data["step"]
            ans = step_data["uncertainty"]["majority_answer"] or step_data["answer"]
            is_correct = answers_match(ans, gold)
            step_correct[step_idx].append(is_correct)
            step_answers[step_idx].append({
                "question": r["question"][:80],
                "predicted": ans,
                "gold": gold,
                "correct": is_correct,
                "agreement": step_data["uncertainty"]["agreement"],
            })

    accuracy_per_step = {}
    for step_idx in sorted(step_correct.keys()):
        correct_list = step_correct[step_idx]
        acc = sum(correct_list) / len(correct_list) if correct_list else 0.0
        accuracy_per_step[step_idx] = {
            "accuracy": round(acc * 100, 2),
            "num_examples": len(correct_list),
            "num_correct": sum(correct_list),
        }

    # ── 2. Average number of calls ───────────────────────────────
    total_calls_list = [r["total_calls"] for r in results]
    num_steps_list = [r["num_steps"] for r in results]
    avg_calls = np.mean(total_calls_list)
    avg_steps = np.mean(num_steps_list)
    early_stop_rate = sum(1 for r in results if r["stopped_early"]) / len(results)

    # ── 3. Final accuracy (last step per example) ────────────────
    final_correct = []
    for r in results:
        gold = r["gold_answer"]
        is_correct = answers_match(r["final_answer"], gold)
        final_correct.append(is_correct)
    final_accuracy = sum(final_correct) / len(final_correct) if final_correct else 0.0

    # ── 4. False-certainty cases ─────────────────────────────────
    # High agreement (>= threshold) but wrong final answer
    false_certainty = []
    for r in results:
        gold = r["gold_answer"]
        last_step = r["steps"][-1]
        agreement = last_step["uncertainty"]["agreement"]
        final_ans = r["final_answer"]
        is_correct = answers_match(final_ans, gold)

        if agreement >= 0.8 and not is_correct:
            false_certainty.append({
                "question": r["question"],
                "final_answer": final_ans,
                "gold_answer": gold,
                "agreement": agreement,
                "num_steps": r["num_steps"],
            })

    # ── 5. Uncertainty drops but still wrong ─────────────────────
    uncertainty_drop_wrong = []
    for r in results:
        gold = r["gold_answer"]
        steps = r["steps"]
        if len(steps) < 2:
            continue

        # Check if uncertainty decreased (agreement increased) across steps
        first_agreement = steps[0]["uncertainty"]["agreement"]
        last_agreement = steps[-1]["uncertainty"]["agreement"]
        final_ans = r["final_answer"]
        is_correct = answers_match(final_ans, gold)

        if last_agreement > first_agreement and not is_correct:
            uncertainty_drop_wrong.append({
                "question": r["question"],
                "final_answer": final_ans,
                "gold_answer": gold,
                "agreement_first_step": first_agreement,
                "agreement_last_step": last_agreement,
                "num_steps": r["num_steps"],
            })

    # ── Build summary ────────────────────────────────────────────
    summary = {
        "num_examples": len(results),
        "final_accuracy_pct": round(final_accuracy * 100, 2),
        "accuracy_per_step": accuracy_per_step,
        "avg_total_calls": round(avg_calls, 2),
        "avg_num_steps": round(avg_steps, 2),
        "early_stop_rate_pct": round(early_stop_rate * 100, 2),
        "false_certainty_count": len(false_certainty),
        "uncertainty_drop_but_wrong_count": len(uncertainty_drop_wrong),
    }

    # ── Save outputs ─────────────────────────────────────────────
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "false_certainty_cases.json"), "w") as f:
        json.dump(false_certainty, f, indent=2)

    with open(os.path.join(output_dir, "uncertainty_drop_wrong.json"), "w") as f:
        json.dump(uncertainty_drop_wrong, f, indent=2)

    with open(os.path.join(output_dir, "full_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Print report ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ITERATIVE AUDIT LOOP — EVALUATION REPORT")
    print("=" * 65)
    print(f"  Examples evaluated:        {len(results)}")
    print(f"  Final accuracy:            {summary['final_accuracy_pct']}%")
    print(f"  Avg total LLM calls:       {summary['avg_total_calls']}")
    print(f"  Avg audit steps:           {summary['avg_num_steps']}")
    print(f"  Early stop rate:           {summary['early_stop_rate_pct']}%")
    print("-" * 65)
    print("  Accuracy per step:")
    for step_idx, info in accuracy_per_step.items():
        role = "Proposer " if step_idx == 0 else f"Auditor {'B' if step_idx % 2 == 1 else 'A'}"
        print(f"    Step {step_idx} ({role}): "
              f"{info['accuracy']}%  ({info['num_correct']}/{info['num_examples']})")
    print("-" * 65)
    print(f"  False-certainty cases:     {len(false_certainty)}")
    print(f"  Uncertainty ↓ but wrong:   {len(uncertainty_drop_wrong)}")
    print("=" * 65)

    if false_certainty:
        print("\n  📌 Sample false-certainty case:")
        fc = false_certainty[0]
        print(f"     Q: {fc['question'][:100]}...")
        print(f"     Predicted: {fc['final_answer']}  |  Gold: {fc['gold_answer']}")
        print(f"     Agreement: {fc['agreement']}")

    if uncertainty_drop_wrong:
        print("\n  📌 Sample uncertainty-drop-but-wrong case:")
        ud = uncertainty_drop_wrong[0]
        print(f"     Q: {ud['question'][:100]}...")
        print(f"     Predicted: {ud['final_answer']}  |  Gold: {ud['gold_answer']}")
        print(f"     Agreement: {ud['agreement_first_step']} → {ud['agreement_last_step']}")

    print(f"\n  Results saved to: {output_dir}/")
    print()

    return summary