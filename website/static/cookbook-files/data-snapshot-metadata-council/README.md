# Running example files

The images are real Data Snapshots. The two Discovery references are published human-reviewed semantic annotations. The numbered Markdown, JSON and JSONL files describe one illustrative round that has never been executed. The role instructions describe the proposed workflow.

- Use the sample register to resolve the exact image bytes and the Discovery reference. Its paths are relative to the site's static root.
- P001 and S001 are inputs to round 001. Its extraction records cite P001. P002 and S002 are outputs of round 001 and would become inputs to the next round.
- Review the illustrative extraction and judge files together. Each line is one record keyed by sample ID. The metadata payload uses Schema v1.4. The surrounding fields are experiment bookkeeping.
- Use the manifest to find all eight artifact types. The local set also includes sample registration and role instructions as supporting materials.

## Attribution and reuse

D1 and C1 come from the policy brief *From Stateless to Citizens: The Journey of the Shona Community in Kenya*, credited to IFRC, IOM, and REACH in the [ReliefWeb source metadata](https://reliefweb.int/report/kenya/policy-brief-stateless-citizens-journey-shona-community-kenya). D1 is the original corpus crop. C1 reuses the narrower production crop from the Data Snapshot end-to-end cookbook. The corpus table with the same basename has different boundaries. D2 comes from World Bank Policy Research Working Paper 5070 by Mafalda Duarte, Outi Korkeala, and David Newhouse. C2 comes from Working Paper 4771 by Thorsten Beck, Leora Klapper, and Juan Carlos Mendoza. The sample register records the source links, PDF pages, asset checksums, and dataset revisions.

The snapshot dataset publishes its MIT terms with a World Bank IGO rider at https://huggingface.co/datasets/ai4data/data-snapshot/blob/626f3c54ba6d8f6e0ba0795f9e544e24b2650faf/LICENSE. The semantic gold dataset releases the annotation JSON under MIT. Preserve source credits and applicable source-publication terms when reusing crops. The gold license covers the metadata release and gives no additional permission to reuse publisher images.

## Context separation

These files are organized for human readers. An experiment runner must construct a separate packet for each model role. In particular, a council session must receive neither the full Discovery references nor filesystem or network tools that can retrieve them. Provide opaque image identifiers to model roles. Source URLs and file hashes belong in the audit record.
