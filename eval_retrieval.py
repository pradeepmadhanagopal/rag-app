from rag import retrieve_with_scores

TEST_CASES = [ {"question": "what is max pooling?", "must_contain": "max pooling"},
    {"question": "why do conv layers share weights across the image?", "must_contain": "same weights"},
    {"question": "what did Hubel and Wiesel discover?", "must_contain": "visual cortex"},
    {"question": "how does stride affect the output size?", "must_contain": "stride"},
    {"question": "what is Q-learning?", "must_contain": None},
    {"question": "explain STRIPS planning", "must_contain": None},
]

def run_evals(k: int = 3, threshold: float = 0.65):
    hits, total = 0, len(TEST_CASES)

    for case in TEST_CASES:
        
        results = retrieve_with_scores(case["question"], k=k, threshold=threshold)

        if case["must_contain"] is None:
            # Negative case: pass = empty retrieval
            passed = len(results) == 0
        else:
            # Positive case: pass = expected text appears in a retrieved chunk
            passed = any(case["must_contain"].lower() in chunk.lower() for chunk, _ in results)

        hits += passed
        status = "PASS" if passed else "FAIL"
        top = f"{results[0][1]:.3f}" if results else "—"
        print(f"[{status}] top_score={top}  {case['question']}")

    print(f"\nHit rate: {hits}/{total} = {hits/total:.0%}  (k={k}, threshold={threshold})")


if __name__ == "__main__":
    run_evals()