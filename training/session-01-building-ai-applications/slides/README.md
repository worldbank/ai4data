# Slides

- `deck.pptx`: the deck, 20 slides.
- `outline.md`: the text of every slide with its demonstration pointer and speaker notes. This is the file to review and comment on.
- `build_deck.js`: the source of both. Edit the script, then regenerate.
- `canva.json`: the public view link of the deck presented in the session, which is kept in Canva. The docs site shows it on the session's Slides page in Canva's embedded viewer.

## Regenerating

```bash
npm install pptxgenjs          # once, in any folder; then point NODE_PATH at its node_modules
NODE_PATH=/path/to/node_modules node build_deck.js
```

The deck uses two fonts that ship with Office, Cambria for titles and Calibri for text, so it renders the same on any machine.

## Slide plan

| Slides | Topic | Demonstration |
|---|---|---|
| 1 to 3 | Title, objectives, the worked example | |
| 4 | What makes a model an application | |
| 5 to 6 | Instructions and context; why context is selected | Notebook 01 |
| 7 | Preparing knowledge | Notebook 02 |
| 8 to 9 | Retrieval; retrieval and generation as two problems | Notebook 03 |
| 10 to 11 | Tools; structured outputs and MCP | Notebook 04 |
| 12 | Which technique solves which problem | |
| 13 to 14 | Workflows and agents; choosing between them | Notebook 05 |
| 15 to 17 | Verification; evaluation; four checks | Notebook 06 |
| 18 | Data360 Chat | Live demonstration |
| 19 to 20 | Recap; resources and questions | |
