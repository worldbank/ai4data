# Role instructions for the proposed workflow

Evidence status: proposed templates. Apply these contracts in separate, scoped contexts.

## Reference extractor

Receive one image, the current policy, and the fixed output schema. Produce one semantic metadata record. The runner attaches the opaque sample ID and versions outside the model response. The runner records validation errors separately from semantic findings.

## Diagnostic judge

Receive the image, extraction, executed policy, and schema. For Discovery, add the entire corresponding human-reviewed annotation. For Challenge, mark reference comparison unavailable.

Check image support, completeness, and compliance separately. Use the fixed assessment rubric. Record the affected field, a precise visual evidence location, the applicable rule if any, uncertainty, and escalation status. Distinguish a likely extraction error, policy violation, ambiguity, and suspected gold or schema issue. Send field-level findings that a council member can inspect against the image. Proposals for policy changes belong to the council.

## Human reviewer

Inspect escalations and a documented selection of cases the judge did not flag. Compare the image, current policy and schema, output, and full reference where available. Record the disposition, rationale, and any authorized change to the gold or schema version. Share scoped decision summaries with deliberation roles. A proposal from another role is input to this decision.

## Council member

Receive the image, extraction, current policy and state, diagnostic findings, and relevant scoped human decisions. Inspect the evidence directly. Submit an independent proposal with the finding ID, suggested rule operation, rationale, and counterexample to check. After independent submissions close, critique the other proposals and revise the recommendation. Raise gold and schema concerns for human review.

## Orchestrator

Synthesize the proposals, critiques, revisions, and scoped evidence. Produce the next policy, cumulative state, and round findings. Cite the finding and disposition behind each accepted change. Preserve the history of rejected and unresolved proposals. Record gold and schema questions for the human reviewer and carry forward only authorized revisions.
