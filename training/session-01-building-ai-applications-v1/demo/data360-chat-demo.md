# Data360 Chat demonstration

Five minutes at the end of the session, after notebook 06. The demonstration shows the concepts of the six notebooks in the application the program runs: retrieval over a catalog, data grounding through MCP tools, citations, and verified numbers.

## Before the session

- [ ] Open the chat application and confirm that it answers; note the model and the MCP server it is connected to.
- [ ] Run every query below and paste the observed behaviour into the "observed" column. A query that fails is kept with its failure noted, or replaced.
- [ ] Take a screenshot of each result for the fallback deck (`fallback/` folder, one PNG per query, no names in the captures).
- [ ] Confirm that no query returns confidential or internal data.

## Queries

| # | Query | Shows | Expected behaviour | Observed (fill in) |
|---|---|---|---|---|
| 1 | How have internet access and economic development evolved in Rwanda and Kenya? Compare the recent available data and cite the sources. | The worked example on the live system | Searches the catalog, calls the data tool for the series, answers with the latest years and sources; numbers carry verification marks | |
| 2 | What share of Kenya's population used the internet in 2025? | Missing data | States that the latest available year is 2024 and gives it | |
| 3 | How connected is Kenya? | Ambiguity | Asks which measure, or names the indicator it chose | |
| 4 | Which indicator should I use to compare internet access across countries, and how is it defined? | Metadata retrieval | Returns the indicator with its definition and source | |
| 5 | What was Rwanda's GDP per capita in 2023, in current US dollars? | Data grounding and verification | One number, with the source and a verification mark | |

## What to point at during the demonstration

1. **The tool calls.** Open the trace or the tool-call view, if the interface exposes it, and name the tools: search, metadata, data. Say that these are the Data360 MCP server's tools and that any MCP client can call them.
2. **The citation.** Show where the number came from: the indicator, the country, the year, the source.
3. **The mark beside the number.** Say what verified, flagged, and unmarked mean, and that an unmarked number receives no mark by design.
4. **One limitation.** Pick one from the catalogue entry of the Data360 MCP server: search over the catalog can miss a series, and a dimension listed in metadata may have no data. Say how the evaluation suite catches it.

## Engineering choices to mention

Three choices from the application that connect to the notebooks:

- Tools over the catalog, for the same reason as notebook 04: numbers come from the data.
- A bounded loop with a trace, as in notebook 05, so that an answer can be inspected.
- Claim tags and a verifier in the renderer, as in notebook 06, with the Proof-Carrying Numbers packages.

## Fallback

If the application or the network is unavailable, show the screenshots in `fallback/` in the same order and narrate the same four points. The notebooks' cached outputs cover the same behaviours offline.

## Links

- Data360 MCP server: https://github.com/worldbank/data360-mcp
- Data AI Chatbot: https://github.com/worldbank/data-ai-chatbot
- Proof-Carrying Numbers: https://worldbank.github.io/ai4data/docs/pcn/
