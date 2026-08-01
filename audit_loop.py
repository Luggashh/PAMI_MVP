"""
Core iterative audit loop with explicit Chain-of-Thought prompting.

At every step, the model is instructed to:
    1. Think through the problem step by step
    2. Label each reasoning step clearly
    3. State the final answer in a parseable format

The full CoT trace is preserved in the results for analysis.
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
    SAVE_COT,
)

# ── System Prompts with explicit CoT instructions ────────────────

PROPOSER_SYSTEM = """You are an expert mathematician. Solve the given math word problem by thinking step-by-step.

You MUST wrap your response exactly in these XML tags:

<reasoning>
Step 1: [Identify facts and requirements]
Step 2: [Break down the arithmetic steps]
...
</reasoning>

<verification>
[Double-check your calculations here]
</verification>

<final_answer>[Insert only the final numerical value here]</final_answer>

Do not put any text or explanations outside of these tags. Provide only the final number inside <final_answer>."""

AUDITOR_SYSTEM = """You are a critical math auditor. Your task is to verify if the proposed solution to the math problem is correct. Do not blindly trust it. Check every calculation.

You MUST wrap your response exactly in these XML tags:

<audit>
Step 1: [Verify the first step of the proposed solution]
Step 2: [Verify the next step]
...
</audit>

<errors>
[List any arithmetic or logical errors found, or write "None"]
</errors>

<corrected_reasoning>
[If errors were found, provide the full step-by-step correct calculation here. Otherwise, repeat the correct steps.]
</corrected_reasoning>

<final_answer>[Insert only the final verified numerical value here]</final_answer>

Do not put any text or explanations outside of these tags. Provide only the final number inside <final_answer>."""

def run_audit_loop(question: str) -> dict:
    """
    Run the full iterative audit loop for a single GSM8K question.

    Returns:
        Dict with:
            - question: the original question
            - steps: list of step dicts, each containing full CoT traces
            - final_answer: the last answer produced
            - total_calls: total number of LLM calls
            - stopped_early: whether we stopped due to high agreement
            - chain_of_thought: ordered list of CoT traces across all steps
    """
    steps = []
    total_calls = 0
    previous_response = None
    chain_of_thought = []  # Collects CoT across all steps

    for step_idx in range(MAX_STEPS):
        is_proposer = step_idx == 0
        role = "proposer" if is_proposer else ("auditor_B" if step_idx % 2 == 1 else "auditor_A")

        if is_proposer:
            system_prompt = PROPOSER_SYSTEM
            prompt = f"Problem:\n{question}\n\nSolve this step by step using the XML format."
        else:
            system_prompt = AUDITOR_SYSTEM
            prompt = (
                f"Problem:\n{question}\n\n"
                f"Proposed solution to audit:\n{previous_response}\n\n"
                f"Carefully audit every step of this solution. Check each calculation for errors. "
                f"Output your analysis and the correct final number using the XML format."
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

        # ── Build CoT trace for this step ────────────────────────
        cot_entry = {
            "step": step_idx,
            "role": role,
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "chain_of_thought": response,  # Full CoT response
            "extracted_answer": main_answer,
            "uncertainty_samples": [],
        }

        # Store CoT from each uncertainty sample too
        if SAVE_COT:
            for sample_idx, (sample_text, sample_answer) in enumerate(
                zip(uncertainty["samples"], uncertainty["answers"])
            ):
                cot_entry["uncertainty_samples"].append({
                    "sample_idx": sample_idx,
                    "chain_of_thought": sample_text,
                    "extracted_answer": sample_answer,
                })

        chain_of_thought.append(cot_entry)

        # ── Step data (same as before, plus CoT) ─────────────────
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

        # ── Early stopping ───────────────────────────────────────
        if uncertainty["agreement_count"] >= AGREEMENT_THRESHOLD:
            break

    last_step = steps[-1]
    final_answer = last_step["uncertainty"]["majority_answer"] or last_step["answer"]

    return {
        "question": question,
        "steps": steps,
        "final_answer": final_answer,
        "num_steps": len(steps),
        "total_calls": total_calls,
        "stopped_early": len(steps) < MAX_STEPS,
        "chain_of_thought": chain_of_thought,  # Full CoT trace
    }
