# Microdata access policy (template)

**Organization:** [name]
**Approved by:** [role]
**Version:** [number], [date]
**Applies to:** all microdata files catalogued by the organization

## 1. Access tiers

| Tier | Who | Conditions | Disclosure control applied |
|---|---|---|---|
| Public use file | Anyone, after agreeing to the terms of use | Registration optional; citation required | Direct identifiers removed; geography limited to [level]; top-coding of [variables]; [other] |
| Licensed file | Registered users with an approved application | Signed agreement; stated purpose; no re-identification; no onward transfer | As public use file, with finer geography or additional variables as listed in the catalog record |
| Secure access | Approved researchers | Access in a secure environment or through remote execution; output checking | Pseudonymized; outputs checked before release |

## 2. Statement in every catalog record

Each study record states its tier in `study_desc.data_access` (NADA: the
access type), the citation requirement, and the restrictions, so that the
conditions travel with the metadata and any system reading the record can
apply them.

## 3. Automated and AI access

- Only public use files are available to automated clients, including AI
  agents and crawlers.
- Automated clients receive the same terms of use and citation requirement
  in machine-readable form (licence and citation fields in the record).
- Licensed and secure-access files are never exposed through an API or an
  agent interface. Their metadata (study and variable level) may be exposed,
  without the data.

## 4. Disclosure control

Before any file enters a tier, the disclosure control steps for that tier
are applied and recorded in the study record (`data_files.data_checks` or
the notes field). The organization keeps a register of the steps applied to
each release.

## 5. Review

This policy is reviewed every [interval] and after any incident.
