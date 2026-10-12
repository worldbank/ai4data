# Policy on the use of AI in official statistics

**Organization:** National Statistics Office (example)
**Approved by:** Director General
**Version:** 1.0, 2025-09-15
**Review date:** 2026-09-15

## 1. Scope

This policy covers the use of AI systems, including language models,
embedding models, classification models, and automated agents, in the
production, review, and dissemination of official statistics by the
National Statistics Office. It applies to staff, contractors, and systems
operated on the office's behalf, and to assistants used by staff in
everyday work.

## 2. Principles

AI use in this office follows the Fundamental Principles of Official
Statistics. In particular:

- Published statistics remain the responsibility of the office. AI systems
  propose; staff decide.
- Confidentiality of individual data is maintained in every AI use.
- Methods, including the use of AI, are documented and published.
- A person can explain any published figure without reference to a model.

## 3. Permitted uses

The following uses are permitted, subject to the controls in section 5:

- Drafting metadata (titles, descriptions, keywords) for review by a curator.
- Ranking and retrieval of published catalog records for search.
- Answering user questions from published records and values, with citations
  and every number verified against the published value.
- Coding open-text survey responses to standard classifications, with a
  confidence threshold and a re-coded sample decided by the methodology unit.
- Explaining edit failures and anomalies to editors and analysts, who decide
  every case.
- Drafting release commentary from output tables, with every number verified
  and an analyst's approval before publication.
- Classifying and extracting information from published documents.

## 4. Prohibited uses

- Sending confidential, unpublished, or embargoed data to any external AI
  service.
- Publishing AI-generated numbers that have not been verified against the
  official record.
- Publishing AI-generated text as official metadata or commentary without
  named human approval.
- Letting any AI component change data, metadata, or estimates directly.
- Using AI to make decisions about individuals, respondents, or staff.

## 5. Controls

- **Register.** Every AI component is listed in the AI component register
  with a purpose, owner, model, provider, data used, risk tier, and review
  date before it goes live.
- **Approval.** New uses are approved by the AI governance board before
  deployment, and the decision is recorded in the decision log.
- **Human review.** Generated metadata, commentary, and answers require
  approval or verification by a named staff member, recorded with the date.
- **Data classification.** Data are classified as public, internal, or
  confidential. Only public data may be processed by external services.
- **Access.** AI agents use read-only credentials limited to published data.
- **Logging.** Inputs and outputs of production AI components are logged and
  retained for 24 months.
- **Evaluation.** Components are tested against their evaluation suite before
  deployment and after any change of model, prompt, or data.
- **Incidents.** A component whose output reached users with an error is
  switched off until the incident is closed by the board.
- **Transparency.** The office publishes this policy, the register's public
  columns, and a statement of model use with every release that used a model.

## 6. Roles

| Role | Responsibility |
|---|---|
| Director General | Approves this policy and chairs the AI governance board |
| Head of methodology | Owns the AI-use policy, the register, and the risk register |
| Head of dissemination | Owns the statement of model use and user-facing components |
| Head of IT | Owns the provider register, hosting, and the cost record |
| Legal adviser | Owns licences and terms of use |
| Component owners | Operate their component within this policy and keep its row current |

## 7. Review

This policy is reviewed every twelve months, after any incident, and before
each AI-readiness re-assessment.
