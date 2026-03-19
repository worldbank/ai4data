# Repository structure

This repository is a **multi-language monorepo**: the Python package, multiple JavaScript/TypeScript libraries under the **@ai4data** npm organization, shared docs, and discovery demos all live in one repo with shared versioning.

## Layout

```
ai4data/
├── src/
│   └── ai4data/                    # Python package (pip install ai4data)
├── packages/
│   └── ai4data/                    # @ai4data npm scope – one folder per package
│       ├── core/                   # npm: @ai4data/core
│       │   ├── package.json
│       │   ├── src/
│       │   └── README.md
│       ├── search/                 # npm: @ai4data/search
│       └── <library>/              # npm: @ai4data/<library>
├── package.json                    # Root workspace (workspaces: ["packages/ai4data/*"])
├── docs/
├── notebooks/
├── discovery/
├── pyproject.toml
└── README.md
```

## JavaScript packages (@ai4data)

All JS/TS libraries are published under the **ai4data** npm organization as scoped packages:

| Path                       | npm package                |
|----------------------------|----------------------------|
| `packages/ai4data/core/`   | `@ai4data/core`            |
| `packages/ai4data/search/` | `@ai4data/search`          |
| `packages/ai4data/<name>/` | `@ai4data/<name>`          |

- **Install one package:** `npm install @ai4data/core` or `npm install @ai4data/search`
- **Add a new library:** Create `packages/ai4data/<library>/` with its own `package.json` where `"name": "@ai4data/<library>"`. It is included in the root workspace automatically.
- **Root workflow:** From repo root, `npm install` installs all workspace packages; `npm run build` and `npm run test` run in all workspaces (each package can define its own scripts).

## Rationale

- **`packages/ai4data/<library>/`** – One directory per npm package; the path mirrors the scope and name (`@ai4data/<library>`).
- **Root `package.json`** – `"workspaces": ["packages/ai4data/*"]` lets you install, build, and test all JS packages from the root and link them locally.
- **Single version** – You can use one version (e.g. from git tags) for all @ai4data packages and the Python package when releasing.

## Versioning and releases

- **Python:** PyPI via existing workflow (tag → build → publish).
- **JavaScript:** Each `packages/ai4data/<library>/` can be published to npm as `@ai4data/<library>`. In CI you can:
  - Run `npm install` and `npm run build` (and test) for all workspaces.
  - On release tags, publish changed packages (e.g. with [changesets](https://github.com/changesets/changesets) or a script that runs `npm publish` in each package directory with the right version).

### Publishing @ai4data/search to npm

Maintainers with publish access to the **ai4data** npm org can release the search package:

1. Bump version in `packages/ai4data/search/package.json` (or run `npm version patch|minor|major` from that directory).
2. From `packages/ai4data/search`, run: `npm publish --access public`. The `prepublishOnly` script builds the package first.
3. Optionally create a git tag (e.g. `@ai4data/search@1.0.0`) and push.

See [packages/ai4data/search/README.md](../packages/ai4data/search/README.md#publishing-maintainers) for full publishing steps and prerequisites.

## Adding a new @ai4data library

1. Create `packages/ai4data/<library>/` with:
   - `package.json` with `"name": "@ai4data/<library>"` and `repository.directory`: `"packages/ai4data/<library>"`
   - `src/`, tests, and a README
2. Run `npm install` at the repo root so the new package is linked in the workspace.
3. Optionally add a short note in the main README listing the new package.

No change to the root `package.json` is needed; `packages/ai4data/*` already includes any new folder under `packages/ai4data/`.
