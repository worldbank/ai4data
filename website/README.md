# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

```bash
npm install
```

**Note**: feel free to use the package manager of your choice.

## Local Development

```bash
npm run start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Changing or removing a page

Every page route the site has published is listed in `published-routes.txt`, and each one keeps working after a page moves. A plugin, `plugins/published-routes.mjs`, checks this after every build.

- **Moving or renaming a page.** An entry in `redirects.json` maps the old route to the new one, for example `{"from": "/training/session-01/documents.html", "to": "/training/session-01/documents"}`. Routes are written without the base URL. The build writes a redirect page at the old route, and the redirect keeps any `#anchor` of the old link.
- **Adding a page.** After `npm run build`, `npm run routes:update` adds the routes of the build to `published-routes.txt`, and the updated file is committed with the change.
- **Retiring a page without a successor.** Its line is deleted from `published-routes.txt` in the same pull request, so that the removal is visible in review.

The build fails when a published route has neither a page nor a redirect, and when a redirect points to a route that is not a page. A new route that is missing from `published-routes.txt` is a warning in a local build and an error in CI. Pull requests run the same build in the "Publish to GitHub Pages" workflow, without deploying.

The local preview server (`npm run serve`) rewrites addresses that end in `.html`, so redirects from `.html` routes are tested with a plain static server, for example `python3 -m http.server` in a folder where `ai4data` links to `build`.

## Deployment

Using SSH:

```bash
USE_SSH=true npm run deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub Pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
