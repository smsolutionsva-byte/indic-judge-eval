# Rubric v1 (shared by human labelers and the LLM judge)

Each criterion is scored 1–4. Read the passage, the question and the reference answer first.

## correctness — does the answer give the right fact?
- 4: Matches the reference answer (paraphrase is fine).
- 3: Mostly right; a minor detail is off or missing.
- 2: Partly right; the key fact is wrong or only loosely related.
- 1: Wrong, or does not answer the question.

## faithfulness — is everything stated supported by the passage?
- 4: Every claim is supported by the passage.
- 3: One small unsupported but harmless addition.
- 2: Notable unsupported claims mixed with supported ones.
- 1: Mainly invented / contradicts the passage.

## fluency — is it natural, grammatical text in the question's language?
- 4: Natural; reads like a fluent native writer.
- 3: Understandable with minor awkwardness or a stray word in another language.
- 2: Clearly unnatural, frequent errors, or heavy code-mixing.
- 1: Wrong language or largely ungrammatical.

## completeness — does it cover everything the question asks?
- 4: Fully answers every part of the question.
- 3: Answers the main part; a secondary part is thin.
- 2: Answers only a small part.
- 1: Does not address the question.
