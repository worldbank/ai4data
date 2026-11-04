# Role instructions for the proposed workflow

Evidence status: proposed templates. Apply these contracts in separate, scoped contexts.

## Reference extractor

Receive one image, the current policy, and the fixed output schema. Produce one semantic metadata record. The runner attaches the opaque sample ID and versions outside the model response. The runner records validation errors separately from semantic findings.

## Diagnostic judge

Receive the image, extraction, executed policy, and schema. For Discovery, add the entire corresponding human-reviewed annotation. For Challenge, mark reference comparison unavailable.

Use the same rubric for Discovery and Challenge.

| Check | What to record |
|---|---|
| Evidence support | Whether the image supports the value and its assigned meaning, with a precise visual location. |
| Metadata completeness | Relevant omissions within the review scope. Distinguish schema requirements, relevant optional fields, and information the crop cannot establish. |
| Policy compliance | How the output relates to the executed policy. Cite applicable rule IDs. |
| Uncertainty | What the image leaves unresolved and, if known, what evidence would settle it. |
| Human escalation | Whether a consequential ambiguity, unresolved reference dispute, or proposed schema change needs adjudication before proceeding. |

Name the fields examined in `review_scope`. Each finding has an ID, kind, field, evidence, assessment, applicable `rule_ids`, `uncertainty`, `human_attention`, and `attention_reason`. Use a short description for `uncertainty` when a question remains open and null when the evidence resolves the issue in scope. Use the same convention for both batches. On Discovery, investigate differences with the reference against the image. A difference can point to a problem with the reference itself.

Use `attention_reason` to explain either routing decision. The value `human_attention: false` means the finding can proceed without immediate human adjudication. It does not indicate that the extraction is correct. Send scoped findings that the council can inspect against the image. Reviewers still spot-check selected cases without an escalation. Proposals for policy changes belong to the council.

## Human reviewer

Inspect escalations and a documented selection of cases the judge did not flag. Compare the image, current policy and schema, output, and full reference where available. Record the disposition, rationale, and any authorized change to the gold or schema version. Share scoped decision summaries with deliberation roles. A proposal from another role is input to this decision.

## Council member

Receive the image, extraction, current policy and state, diagnostic findings, and relevant scoped human decisions. Inspect the evidence directly. In each independent proposal, cite the finding and evidence, name the failure mode, and specify whether to add, amend, retain, or retire a rule. Identify the rule or mark it as new. Explain the expected benefit and identify a risk or counterexample to test. After independent submissions close, critique the other proposals and explain what changed in the revised recommendation. Raise gold and schema concerns for human review.

## Orchestrator

Synthesize the proposals, critiques, revisions, and scoped evidence. Produce the next policy, cumulative state, and round findings. Cite the finding and disposition behind each accepted change. Preserve the history of rejected and unresolved proposals. Record gold and schema questions for the human reviewer and carry forward only authorized revisions.
