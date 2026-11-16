# Extraction policy P002

Evidence status: illustrative. This candidate policy is the output of round 001. Its first application would occur in a subsequent round.

Apply the active rules to the supplied image and return a single Data Snapshot Metadata Schema v1.4 record.

- **R01** Use the snapshot image as the evidence boundary. Leave unsupported fields unset or null where the schema permits. Exclude filenames, parent-document context, and outside knowledge from metadata evidence.
- **R02** Use an identifiable visible title or caption for the title. Treat axes, legends, and group headings according to their visual roles. Record a printed figure or table identifier separately as the document label.
- **R03** Record measured concepts, groupings, and statistical forms in their schema fields. Attach a statistic such as a coefficient or standard error to the concept it describes when the layout identifies that association. Preserve distinct lag and specification headings as groupings when supported.
- **R04** Preserve printed wording where the schema asks for source text. Describe the observations without transcribing numerical data.
- **R05** Associate a unit only with concepts for which the snapshot supports that unit. A percent symbol in one row or an unlabeled integer scale alone provides insufficient evidence for other measures or for the counted entity.
