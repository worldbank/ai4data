# Extraction policy P001

Evidence status: illustrative. This is the initial policy for round 001.

Apply the active rules to the supplied image and return a single Data Snapshot Metadata Schema v1.4 record.

- **R01** Use the snapshot image as the evidence boundary. Leave unsupported fields unset or null where the schema permits. Exclude filenames, parent-document context, and outside knowledge from metadata evidence.
- **R02** Record a visible title and document label when present.
- **R03** Record measured concepts, groupings, and statistical forms in their schema fields.
- **R04** Preserve printed wording where the schema asks for source text. Describe the observations without transcribing numerical data.
