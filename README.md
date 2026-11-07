# Indic LLM-Judge Agreement Eval

**Question:** can an LLM judge be trusted to grade Hindi and Kannada answers the way humans do?

**Short answer:** mostly yes on correctness, faithfulness and completeness, but not for fluency.
That held for all 190 answers, which one human labeled against the judge. Quadratic-weighted κ was 0.83 overall
and 0.68 for fluency. Many of the largest disagreements came from errors in the dataset's own reference answers,
not from the judge.

Every number below comes from run `full-20261107-132628` (config, prompt and outputs are in that folder).

> **One human labeller.** All 190 human labels come from a single person (the author). That means there is
> no human–human agreement figure yet, so the results compare the judge with one person, not with a measured
> human ceiling. The labelling tool already supports a second labeller on an overlap set (see Method).

## Dataset

| Language | Items | Source | License |
|---|---|---|---|
| Hindi (hi) | 80 | [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA) (AI4Bharat) | CC-BY-4.0 |
| Kannada (kn) | 80 | IndicQA | CC-BY-4.0 |
| English (en, control) | 30 | [SQuAD v1.1 dev](https://huggingface.co/datasets/rajpurkar/squad) | CC-BY-SA-4.0 |

`python -m indic_judge.cli build-data` downloads the raw files and keeps only answerable
questions with passages of 2,000 characters or fewer. It then samples with seed 13 and
writes `data/items.jsonl`, where each row records its source and license.

## Rubric

There are four criteria, each scored 1–4 with written anchors: **correctness**, **faithfulness**, **fluency**
(natural in the question's language) and **completeness**. See [prompts/rubric.md](prompts/rubric.md).
Humans and the judge use the same rubric text.

## Models

- Generator: `qwen-3.8-27b` through the BharatCode chat-completions API, temperature 0, `max_tokens` 2000.
  It is a reasoning model, so its hidden reasoning counts toward that limit.
- Judge: the same `qwen-3.8-27b`, temperature 0, `max_tokens` 3000, with JSON output validated by pydantic.
  The prompt is in [prompts/judge_v1.md](prompts/judge_v1.md) and is copied into every run folder.
- The configs also support Anthropic and Ollama (`qwen2.5:7b`, optional). Neither was used in the reported run:
  there was no Anthropic key, and Ollama wasn't installed.

## Method

1. Every LLM call is cached in `data/cache/llm_cache.jsonl`. The cache key covers the provider, model, prompt,
   max_tokens and temperature, so reruns cost nothing and any prompt change misses the cache.
2. Humans label blind in a Streamlit tool (`make label`) and never see judge scores or model names.
   Labeler B gets the first N answers that labeler A labeled, for human–human agreement.
3. Agreement measures: Cohen's κ, quadratic-weighted κ and ordinal Krippendorff's α. Each has a paired
   bootstrap 95% CI (1,000 resamples) and is computed overall, per language, per criterion and per
   language × criterion, with confusion matrices.
4. Bias checks:
   - Spearman correlation of answer length with the judge score and with (judge − human).
   - Drift across judging order (position in a shuffled run).
   - A pairwise A/B test in both orders: the share of cases where the judge's pick changes with slot order.
5. Latency and token counts are measured for every call.

## Results

### Judge vs human, by language (all four criteria pooled; 95% bootstrap CIs, 1,000 resamples)

| Language | n scores | Exact agree | Cohen's κ | Quadratic-weighted κ | Krippendorff's α (ordinal) |
|---|---|---|---|---|---|
| **All** | 752 | 92.0% | 0.48 [0.37, 0.60] | **0.83 [0.73, 0.89]** | 0.56 [0.43, 0.67] |
| Hindi | 316 | 91.1% | 0.40 [0.22, 0.58] | 0.73 [0.51, 0.87] | 0.46 [0.26, 0.64] |
| Kannada | 316 | 90.5% | 0.53 [0.38, 0.66] | 0.89 [0.81, 0.93] | 0.63 [0.47, 0.75] |
| English (control) | 120 | 98.3% | 0.49 [0.00, 1.00] | 0.64 [0.00, 1.00] | 0.50 [−0.01, 1.00] |

### By criterion (all languages)

| Criterion | n | Exact agree | Cohen's κ | Quadratic-weighted κ | Krippendorff's α |
|---|---|---|---|---|---|
| Completeness | 188 | 95.7% | 0.66 [0.40, 0.85] | 0.95 [0.87, 0.98] | 0.73 [0.46, 0.90] |
| Faithfulness | 188 | 96.8% | 0.61 [0.24, 0.87] | 0.92 [0.66, 0.98] | 0.63 [0.24, 0.88] |
| Correctness | 188 | 93.1% | 0.58 [0.35, 0.76] | 0.77 [0.53, 0.92] | 0.66 [0.41, 0.83] |
| **Fluency** | 188 | 82.4% | 0.26 [0.08, 0.43] | **0.68 [0.36, 0.81]** | 0.35 [0.13, 0.53] |

`agreement.csv` in the run folder has the full language × criterion table, and the dashboard shows the confusion matrices.
Fluency is the weakest cell in both Indic languages:

- Hindi fluency: 74.7% exact agreement, Cohen's κ 0.16 [−0.05, 0.39].
- Kannada fluency: 83.5% exact agreement, Cohen's κ 0.34 [0.04, 0.59].

**How to read these numbers:** most scores are 4, so exact agreement is high while Cohen's κ stays modest.
This is the kappa paradox. The English control shows it most clearly: agreement is 98%, but the CIs run from 0
to 1, and some English cells have undefined κ because both raters gave every answer a 4.

### Bias checks (Spearman ρ, n = 190)

| Check | ρ |
|---|---|
| Answer length vs judge mean score | −0.01 |
| Answer length vs human mean score | −0.04 |
| Answer length vs (judge − human) | 0.01 |
| Judging order position vs judge score | −0.05 |

There is no detectable length bias or drift with judging order. The pairwise A/B position test did not run,
because it needs two generators and only one was available.

### Reliability, latency and tokens

- Judge parse failures: 2 of 190 (1.1%). In both (`hi-048`, `kn-120`) the judge used its whole 3,000-token limit
  before it wrote the JSON.
- Empty answers: 3 of 190 (`hi-031`, `kn-139`, `kn-156`). The generator spent all 2,000 tokens on reasoning.
- Judge latency per judgment: 11.5 s mean, 8.9 s median, 25.4 s p95.
- Judge tokens per judgment: 1,583 in and 247 out on average.
- The full run made 380 calls and used 587,086 tokens. It took 832 s of wall-clock time, with 6 parallel calls.
- Cost in USD: not reported, because BharatCode pricing wasn't available, so only tokens were measured.

## Failure cases

Each case shows the actual text and scores in the order correctness / faithfulness / fluency / completeness.

**1. A wrong reference answer: the judge sided with the passage, the human with the reference (`hi-018`).**
- Q: काकोरी प्रकरण में कितने लोग शामिल थे? (How many people were involved in the Kakori case?)
- Reference: पाँच (five)
- Answer: काकरी प्रकरण में केवल १० ही लोग वास्तविक रूप से शामिल थे। (only 10 people were actually involved)
- Judge: 4/4/4/4. Rationale: "the reference answer of five refers to those who escaped".
- Human: 1/4/3/4. The labeler's note says the passage gives ten.
- The two raters applied different rules: the rubric scores correctness against the reference, but the reference
  is wrong. The labeler's notes flag 12 items where the reference or the question doesn't fit the passage.
  Four of the ten worst disagreements are among them (`hi-018`, `hi-008`, `en-176`, `kn-141`).

**2. The same conflict in the opposite direction (`en-176`).**
- Q: What was the name of the Marco Polo episode?
- Reference: Mission to the Unknown
- Answer: The Marco Polo episode was named "Mission to the Unknown."
- Judge: 1/1/4/4. Rationale: "The passage lists Marco Polo and Mission to the Unknown as separate episodes".
- Human: 4/1/4/4.
- This time the judge followed the passage and penalized the answer, while the human gave credit for matching the reference.
  Neither rater handled reference-vs-passage conflicts consistently, which points to a gap in the rubric.

**3. Garbled Hindi got full correctness (`hi-029`).**
- Q: लाल बहादुर शास्त्री की पत्नी का क्या नाम था? (What was the name of Lal Bahadur Shastri's wife?)
- Answer: लाल बहोदर शोस्रिी की पत्नी को नम ललितो शोस्रिी थो।
- Judge: 4/4/1/4. Human: 3/3/2/4.
- The judge noticed the garbling and gave fluency 1. It still gave full correctness to an answer whose names are
  misspelled beyond recognition.

**4. A partly wrong refusal scored as nearly correct (`kn-141`).**
- Q: ಲಾಹೋರ್‌ನ ಬಾದಶಾಹಿ ಮಸೀದಿಯನ್ನು ಯಾರು ನಿರ್ಮಿಸಿದನು? (Who built Lahore's Badshahi Mosque?)
- Reference: ಶಹಜಹಾನ್ (Shah Jahan)
- Answer: "this passage has no information about Lahore's Badshahi Mosque."
- Judge: 3/4/4/2. Human: 1/4/3/1.
- The judge gave a non-answer correctness 3, even though its own rationale says the passage names Shah Jahan as
  a builder of mosques in Lahore.

**5. The generator ran out of tokens (`kn-139`).**
- Q: ಋಗ್ವೇದದ ಸಮಾಜದಲ್ಲಿ ಜನಪ್ರಿಯ ಪಾನೀಯಗಳಾವುವು? (What were the popular drinks in Rigvedic society?)
- Answer: *(empty)*. All 2,000 output tokens were spent on reasoning.
- Both raters scored it 1/1/1/1.
- The judge handles empty answers correctly. The failure is that the reasoning model needs a larger output budget.

## Limitations

- **One annotator, and no human–human agreement yet.** Labeler B hasn't labeled anything, so there is no
  human-agreement ceiling to compare the judge against.
- **Self-judging:** the same model (`qwen-3.8-27b`) wrote and judged every answer, so self-preference can't be
  ruled out. That would also help explain why most scores are 4.
- **Labeling depth:** the median time between saved labels was about 10 s. That suggests most labels were checked
  against the reference rather than a full reading of the passage, which is especially relevant for faithfulness.
- **Noisy references:** the annotator's notes flag 12 of 190 items where the IndicQA/SQuAD reference or question
  doesn't fit the passage. The
  rubric doesn't say whether correctness follows the reference or the passage, and a v2 rubric should.
- **Ceiling effects:** with a single generator, most scores are 4, so κ and the per-cell CIs are unstable,
  especially for English (n = 30).
- **Small sample:** 190 items. Per-language × criterion cells have 79 or fewer scores, and some CIs span 0–1.
- **No position test:** the pairwise A/B position-bias check needs a second generator.

## Reproduce

```bash
pip install -e ".[dash,dev]"
cp .env.example .env   # then export BHARATCODE_API_KEY in your shell
make eval              # 20-item sample, end to end
make label             # label answers (blind)
python -m indic_judge.cli analyze runs/<run_id>
make dashboard         # or: make docker
```

Without `make` on Windows, run the `python -m indic_judge.cli ...` lines from the Makefile directly.

**Cost guard:** before calling the API, a run estimates the worst-case cost of its uncached calls
(output assumed to be max_tokens) and stops if that exceeds `max_budget_usd` (default $3).
This is a ceiling for gating the run, not a measurement. It only applies to models that have a price set in the config. No
BharatCode price was available, so the guard treated the reported run as $0 and did not limit it. That run's
measured usage is 587,086 tokens. Every response is cached in `data/cache/llm_cache.jsonl`, so reruns make no
API calls.

## Licence

- **Code** (everything under `src/`, `tests/`, `dashboard/`, `configs/`, `prompts/` and the build files):
  [MIT](LICENSE).
- **Data** stays under its original licences, not MIT:
  - The Hindi and Kannada items, and model answers derived from them, come from
    [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA) by AI4Bharat and are licensed **CC-BY-4.0**.
  - The English items come from [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) and are licensed **CC-BY-SA-4.0**.
  - This covers `data/items.jsonl`, the cached responses in `data/cache/`, and the copies in `runs/`. Each item
    records its source and licence.

## Tests

`pytest` covers:

- Metric functions, checked against hand-computed κ, scikit-learn, and the published α values
  from Krippendorff (2011): nominal 0.743, interval 0.849.
- Config loading.
- Judge-output parsing.
- Cache behavior.
- The labeling queue.
- An analysis smoke test that uses synthetic fixtures in a temp directory.
