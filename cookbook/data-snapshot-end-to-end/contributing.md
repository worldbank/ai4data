---
id: contributing
title: Contributing
sidebar_position: 92
description: This page explains how to propose evidence-backed corrections and recipes.
---

# Contributing

Contributions follow the [cookbook authoring guide](/cookbook/authoring/). A recipe needs a defined result, an explanation before instructions, a worked example, verification steps, and a statement of its limits.

## Evidence for a change

A command must trace to the production CLI or API. An executed example must retain its source, settings, and raw outputs. Mock outputs, reviewed gold records, unexecuted instructions, and proposed development must be labelled separately. Credentials must never appear in fixtures or requests committed to the repository.

## Recipe template

```markdown
<Recipe title="N.M Produce a defined result" level="AI-ready" skills="Developer and data curator" time="Indicative time">

**Result:** The organization has an artifact it can inspect.

The recipe explains the artifact before presenting its steps.

1. Prepare the input and record its origin.
2. Run the supported operation.
3. Verify the actual output against the evidence.

**Worked example.** The recipe identifies the real input, action, and output.

**What this does not do.** The recipe states its limits.

</Recipe>
```

## Corrections

Report the affected sentence, source of truth, and a proposed correction through a [GitHub issue](https://github.com/worldbank/ai4data/issues). Changes to the broader document cookbook remain separate from this guide unless specifically agreed with its author.
