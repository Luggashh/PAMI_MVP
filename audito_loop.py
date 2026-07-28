"""
Core iterative audit loop.

Architecture (from project requirements):
    Step 0: Model A proposes an answer.
    Step 1: Model B audits and revises it.
    Step 2: Model A audits again.
    Step 3: Model B audits again.
    ... up to MAX_STEPS total answers, or early stop on high agreement.

At each step, we also sample the current auditor 5 times to compute
majority-vote agreement as an uncertainty proxy.

This design draws on:
- LMvLM (Cohen et al., 2023): using one LM to cross-examine another's claims
- Self-Refine (Madaan et al., 2023): iterative refinement with self-feedback
- The survey by Kamoi et al. (2024) showing that external/cross-model feedback
  is more reliable than pure intrinsic self-correction
"""

from ollama_client import generate
from uncertainty import compute_uncertainty
from utils import extract_numerical_answer
from config import (
    MAX_STEPS,
    AGREEMENT_THRESHOLD,
    UNCERTAINTY_SAMPLES,
    TEMPERATURE_GREEDY,
    SEED_MAIN,
)

# ── System Prompts ───────────────────────────────────────────────

PROPOSER_SYSTEM = (
    "You are a math problem solver. Solve the given math word problem "
    "step by step. Show your reasoning clearly, then state your final "
    "numerical answer on the last line in the format: 'The answer is <number>'."
)

AUDITOR_SYSTEM = (
    "You are a math auditor. You will be given a math word problem and a "
    "proposed solution. Your job is to:\n"
    "1. Carefully check each step of the solution for errors.\n"
    "2. If you find errors, explain them and provide the corrected solution.\n"
    "3. If the solution is correct, confirm it.\n"
    "4. Always state your final numerical answer on the last line in the "
    "format: 'The answer is <number>'."
)

def run_audit_loop(question: str) -> dict:
    """
    Run the full iterative audit loop for a single GSM8K question.

    Returns:
        Dict with:
            - question: the original question
            - steps: list of step dicts (response, answer, uncertainty info)
            - final_answer: the last answer produced
            - total_calls: total number of LLM calls (main + uncertainty samples)
            - stopped_early: whether we stopped due to high agreement
    """
    steps = []
    total_calls = 0
    previous_response = None

    for step_idx in range(MAX_STEPS):
        # Determine role: even steps = Model A, odd steps = Model B
        is_proposer = step_idx == 0
        role = "proposer" if is_proposer else ("auditor_B" if step_idx % 2 == 1 else "auditor_A")

        if is_proposer:
            system_prompt = PROPOSER_SYSTEM
            prompt = f"Problem:\n{question}"
        else:
            system_prompt = AUDITOR_SYSTEM
            prompt = (
                f"Problem:\n{question}\n\n"
                f"Proposed solution:\n{previous_response}\n\n"
                f"Please audit this solution. Check each step for correctness. "
                f"If there are errors, provide the corrected solution with the right answer."
            )

        # ── Main generation (greedy / deterministic) ─────────────
        response = generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=TEMPERATURE_GREEDY,
            seed=SEED_MAIN,
        )
        total_calls += 1

        main_answer = extract_numerical_answer(response)

        # ── Uncertainty estimation (5 stochastic samples) ────────
        uncertainty = compute_uncertainty(
            prompt=prompt,
            system_prompt=system_prompt,
            k=UNCERTAINTY_SAMPLES,
        )
        total_calls += UNCERTAINTY_SAMPLES

        step_data = {
            "step": step_idx,
            "role": role,
            "response": response,
            "answer": main_answer,
            "uncertainty": {
                "answers": uncertainty["answers"],
                "majority_answer": uncertainty["majority_answer"],
                "agreement": uncertainty["agreement"],
                "agreement_count": uncertainty["agreement_count"],
            },
        }
        steps.append(step_data)

        previous_response = response

        # ── Early stopping: high agreement ───────────────────────
        if uncertainty["agreement_count"] >= AGREEMENT_THRESHOLD:
            break

    # Use the majority answer from the last step's uncertainty samples
    # as the final answer (more robust than single greedy pass)
    last_step = steps[-1]
    final_answer = last_step["uncertainty"]["majority_answer"] or last_step["answer"]

    return {
        "question": question,
        "steps": steps,
        "final_answer": final_answer,
        "num_steps": len(steps),
        "total_calls": total_calls,
        "stopped_early": len(steps) < MAX_STEPS,
    }