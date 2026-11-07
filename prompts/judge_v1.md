<!-- judge prompt v1. Changing this file changes the cache key, so old judgments are not reused. -->
You are grading an answer to a reading-comprehension question. The question may be in Hindi, Kannada or English.

{rubric}

Passage:
{context}

Question: {question}

Reference answer: {reference}

Candidate answer: {answer}

Score the candidate answer on each criterion. Reply with ONLY a JSON object, no other text:
{{"correctness": <1-4>, "faithfulness": <1-4>, "fluency": <1-4>, "completeness": <1-4>, "rationale": "<one short sentence in English>"}}
