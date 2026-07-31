CODE_SCORER_PROMPT = """You are an expert clinical coder. Your task is to evaluate whether a candidate ICD-10 code is supported by the provided clinical evidence.

Candidate ICD-10 Code: {icd_code}
Code Title: {icd_title}
Code Description: {icd_description}

Clinical Evidence:
{evidence}

Please evaluate the code and return a JSON object with the following fields:
- "code": The candidate code.
- "supported": true if the evidence supports this code, else false.
- "confidence": Your confidence score between 0.0 and 1.0.
- "evidence_quote": A direct quote from the evidence supporting the code.
- "rationale": A brief explanation of your decision.
- "missing_evidence": Any evidence that would be needed to be more certain.
- "risk_flag": "none", "weak_evidence", "ambiguous", or "possible_hallucination".

Only return valid JSON. Do not include any other text.
"""

BATCH_CODE_SCORER_PROMPT = """You are an expert clinical coder. You are given clinical evidence extracted FROM THE PATIENT'S OWN DISCHARGE NOTE and a list of candidate ICD-10 codes. For EACH candidate, decide whether it is supported by the note evidence.

Candidates (each with the evidence retrieved from the patient's note):
{candidates_block}

Return ONLY a JSON array with one object per candidate, in the same order. Keep it COMPACT:
- "code": the candidate code.
- "supported": true if the note evidence supports this code, else false.
- "confidence": REQUIRED on EVERY object, including when "supported" is false. A number between 0.0 and 1.0.
- "q": ONLY IF supported is true, a direct quote of at most 12 words COPIED from that code's evidence. If supported is false, omit this field entirely.
- "r": ONLY IF supported is true, at most 12 words of justification. If supported is false, omit this field entirely.

Every object MUST contain "code", "supported" and "confidence". Never omit "confidence".

Example of the exact required format for two candidates:
[{{"code":"I509","supported":true,"confidence":0.82,"q":"acute decompensated heart failure","r":"explicitly documented"}},{{"code":"E119","supported":false,"confidence":0.11}}]

Do not output any other fields. Do not repeat the evidence. Only return a valid JSON array.
"""

BATCH_CODE_SCORER_PROMPT_WITH_NOTE = """You are an expert clinical coder. You are given a patient's discharge note and a list of candidate ICD-10 codes. For EACH candidate, decide whether the NOTE supports assigning that code.

DISCHARGE NOTE:
{note}

Candidates (each with a short passage retrieved from the note above):
{candidates_block}

Base your judgement on the FULL note, not only on the retrieved passage: a code may be supported by text the passage does not contain.

Return ONLY a JSON array with one object per candidate, in the same order. Keep it COMPACT:
- "code": the candidate code.
- "supported": true if the note supports this code, else false.
- "confidence": REQUIRED on EVERY object, including when "supported" is false. A number between 0.0 and 1.0.
- "q": ONLY IF supported is true, a direct quote of at most 12 words COPIED from the note. If supported is false, omit this field entirely.
- "r": ONLY IF supported is true, at most 12 words of justification. If supported is false, omit this field entirely.

Every object MUST contain "code", "supported" and "confidence". Never omit "confidence".

Example of the exact required format for two candidates:
[{{"code":"I509","supported":true,"confidence":0.82,"q":"acute decompensated heart failure","r":"explicitly documented"}},{{"code":"E119","supported":false,"confidence":0.11}}]

Do not output any other fields. Do not repeat the note. Only return a valid JSON array.
"""

CONTRASTIVE_VERIFIER_PROMPT = """You are an expert clinical coder. Your task is to distinguish between similar ICD-10 codes based on clinical evidence.

Target Code: {target_code} ({target_title})
Sibling Codes:
{sibling_codes}

Clinical Evidence:
{evidence}

Determine the most appropriate code from the target and sibling codes. Return a JSON object with:
- "preferred_code": The ICD-10 code that best matches the evidence.
- "rejected_codes": A list of objects with "code" and "reason" for why they were rejected.
- "contrastive_rationale": Explanation of why the preferred code is better than the others.
- "confidence": Your confidence score between 0.0 and 1.0.

Only return valid JSON. Do not include any other text.
"""

# ---------------------------------------------------------------------------
# Copy-instructed control (reviewer item T3-5).
#
# The exact-quote compliance rate measures whether the quoted evidence string occurs in the passage
# the scorer was shown. In the primary campaign only 8-21% of quotes did. That number is ambiguous
# on its own: it could mean the model cannot copy, or that it did not understand it was being asked
# to. This variant removes the ambiguity by making the instruction maximally explicit and stating
# the criterion the string will be checked against. The gap between this arm and the primary arm is
# the achievable ceiling of the metric in this harness.
#
# Identical to BATCH_CODE_SCORER_PROMPT except for the "q" field instruction.
# ---------------------------------------------------------------------------
BATCH_CODE_SCORER_PROMPT_COPY_INSTRUCTED = """You are an expert clinical coder. You are given clinical evidence extracted FROM THE PATIENT'S OWN DISCHARGE NOTE and a list of candidate ICD-10 codes. For EACH candidate, decide whether it is supported by the note evidence.

Candidates (each with the evidence retrieved from the patient's note):
{candidates_block}

Return ONLY a JSON array with one object per candidate, in the same order. Keep it COMPACT:
- "code": the candidate code.
- "supported": true if the note evidence supports this code, else false.
- "confidence": REQUIRED on EVERY object, including when "supported" is false. A number between 0.0 and 1.0.
- "q": ONLY IF supported is true. This field is checked by exact string matching against that code's evidence text. COPY a span of at most 12 words CHARACTER FOR CHARACTER from that code's evidence. Do not paraphrase, do not correct spelling, do not expand abbreviations, do not change capitalisation or punctuation, do not join text from different parts of the evidence. If no span of the evidence supports the code, set "supported" to false and omit this field.
- "r": ONLY IF supported is true, at most 12 words of justification. If supported is false, omit this field entirely.

Every object MUST contain "code", "supported" and "confidence". Never omit "confidence".

Example of the exact required format for two candidates:
[{{"code":"I509","supported":true,"confidence":0.82,"q":"acute decompensated heart failure","r":"explicitly documented"}},{{"code":"E119","supported":false,"confidence":0.11}}]

Do not output any other fields. Do not repeat the evidence. Only return a valid JSON array.
"""
