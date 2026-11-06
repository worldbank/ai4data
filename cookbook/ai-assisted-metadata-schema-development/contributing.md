---
id: contributing
title: Contributing
sidebar_position: 92
description: Evidence, scope, and review requirements for contributions to this working draft.
---

# Contributing

This guide is a structural draft. Its candidate recipes establish intended
results and supporting evidence. Procedures, worked examples, executable
exercises, and realistic time estimates remain for later authoring.

Contributions can be proposed through
[GitHub issues](https://github.com/worldbank/ai4data/issues) and pull requests
to the [AI4Data repository](https://github.com/worldbank/ai4data).
The [cookbook authoring guide](/cookbook/authoring/) governs structure and style.

## Criteria for a recipe

- The recipe produces a defined result with an explicit check.
- It names its maturity level, practitioner skills, and indicative time.
- Explanation introduces the metadata concepts before the procedure.
- A worked example uses the prepared evidence and identifies its source and limits.
- Model proposals and human decisions remain distinguishable.
- Code, when added, is imported from its downloadable source and verified on small inputs.

## Evidence and scope

Research results retain their original labels and chronology. A simplified
teaching transformation is identified separately from recorded outputs.
Fresh model executions are replication attempts and keep their outputs
separate. Claims about improvement require comparative evidence.

The original 41-concept assessment belongs to chapter 4. The restoration of
panel_title during operational design belongs to chapter 5. Image reuse
requires appropriate attribution and permission; public availability alone
does not establish redistribution rights.

The guide develops a schema. Production extraction, layout detection, formal
ontology construction, and knowledge graph infrastructure remain outside its scope.

## Recipe template

The following authoring template describes the structure required for future recipes.

```mdx
<Recipe title="N.M Imperative title naming the result" level="Foundational" skills="Role and relevant skills" time="Indicative estimate">

**Result:** State the checkable artifact produced by the recipe.

Explain the artifact, relevant concepts, and purpose before the steps.

1. Describe the action and its reason.
2. Introduce any code or template and explain its fields.
3. State the verification and expected outcome.

**Worked example.** Identify the input, action, output, source, and evidence status.

**What this does not do.** State the limits of the example and procedure.

</Recipe>
```

## Review

Technical review checks evidence fidelity, identities, denominators, links,
and executable behavior where implemented. Editorial review checks conceptual
clarity, the learning progression, human decision rights, and empirical claims.
The standards page identifies reviewed references without implying formal adoption.
