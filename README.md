# ssarangi.github.io

Personal site and blog: systems engineering, with a focus on AI inference infrastructure.
Built with [Astro](https://astro.build) on the [AstroPaper](https://github.com/satnaing/astro-paper) theme,
Tailwind, MDX, Shiki, KaTeX, build-time Mermaid and Pagefind search. Deployed to GitHub Pages by Actions.

## Develop

```bash
npm install
npx playwright install chromium   # one-time; Mermaid diagrams are rendered at build time
npm run dev                       # http://localhost:4321 (drafts are visible in dev)
npm run build                     # astro check + build + Pagefind index (+ broken-link check)
npm run preview                   # serve dist/
```

Search (Pagefind) only works after `npm run build`; in `npm run dev` the search page shows an empty index.

## How to write a post

**Where:** create `src/content/posts/my-post-title.md` (or `.mdx` if you want components). The file name is the URL: `/posts/my-post-title/`.

**Front matter template**

```yaml
---
title: "Constrained decoding without the tears"
description: "One or two sentences. Used in the list, RSS, and social cards."
pubDate: 2026-12-01
updatedDate: 2026-12-10      # optional
tags: ["inference", "decoding"]
draft: true                  # shown in `npm run dev` only; remove to publish
featured: false              # pin to the home page "Featured" list
archived: false              # hide from home/posts/RSS; still listed on /archives
heroImage: ./hero.png        # optional; local file, optimised by Astro
heroImageAlt: "..."
ogImage: ./card.png          # optional; otherwise a card is generated from the title
canonicalURL: https://...    # optional
series: "LoRA serving"       # optional; posts with the same series get prev/next navigation
seriesOrder: 2
hideToc: false
---
```

**Markdown features**

- **Code:** fenced blocks with Shiki highlighting (light and dark themes) and a copy button.
  ` ```python title="parse.py" {2-3} ` gives a filename and highlights lines 2-3.
  Add `// [!code ++]` / `// [!code --]` at the end of a line for diffs, `// [!code highlight]` to highlight a line.
- **Math:** `$inline$` and `$$ block $$` (KaTeX). The KaTeX stylesheet loads only on posts that contain math.
- **Diagrams:** a fenced `mermaid` block is rendered to inline SVG at build time (no client JS, follows dark mode).
- **Footnotes:** `text[^1]` and `[^1]: note`. Headings get anchors; posts with 2+ headings get a table of contents.
- **Images:** `![alt](/img/...)` for files under `public/`. Dimensions and lazy loading are added at build time.

**MDX components** (available in `.mdx` posts without importing):

```mdx
<Callout type="note|warn|finding" title="Optional">Text with **markdown**.</Callout>

<Figure src="/img/blog/x/plot.png" alt="..." caption="Throughput vs batch size" />
<!-- or import hero from "./plot.png" and pass src={hero} to get Astro image optimisation -->

<Aside>A sidenote. It sits in the margin on wide screens.</Aside>

<Chart type="bar|line|scatter" title="Decode throughput" yLabel="tok/s"
  labels={["A", "B"]} datasets={[{ label: "batch 1", data: [92, 118] }]} caption="..." />
<!-- Chart.js, loaded only on pages that use <Chart> -->

<Benchmark
  columns={[{ key: "engine", label: "Engine" },
            { key: "ttft", label: "TTFT", unit: "ms", digits: 1, better: "lower" }]}
  rows={[{ engine: "A", ttft: 84.2 }, { engine: "B", ttft: 61.9 }]} />
<!-- numeric columns are right-aligned with tabular figures; `better` bolds the best value -->
```

See `src/content/posts/hello-again.mdx` (a draft) for every feature in one place.

**Publish:** set `draft: false` (or remove it), commit, and merge to `master`. The GitHub Action builds and deploys.

## Site configuration

- `astro-paper.config.ts`: site URL, title, socials, "edit on GitHub" link, search, TOC.
- `src/site.config.ts`: **Giscus comments** (off; set `enabled: true` plus `repoId`/`categoryId` from <https://giscus.app>)
  and **analytics** (none; set `{ provider: "plausible", domain }` or `{ provider: "goatcounter", code }`).
- `src/data/redirects.json`: old Jekyll URLs to new URLs (generated during migration). `src/data/resume.json`: the résumé page.
- Pages: `src/pages/{projects,now,resume}.astro`, About in `src/content/pages/about.md`.
- The old reveal.js deck lives in `public/presentation/`; legacy images in `public/img/`.

## Deploy

`.github/workflows/deploy.yml` builds with `withastro/action`, checks internal links with lychee, and publishes with `actions/deploy-pages`
on every push to `master` (and manually via **Actions → Deploy → Run workflow**).
Repo setting needed once: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

### Custom domain (optional)

1. Add `public/CNAME` containing the domain (e.g. `example.com`) and set `site.url` in `astro-paper.config.ts`.
2. DNS at your registrar (Cloudflare / Porkbun), apex: `A` records to `185.199.108.153`, `185.199.109.153`, `185.199.110.153`, `185.199.111.153`
   and `AAAA` to `2606:50c0:8000::153`, `2606:50c0:8001::153`, `2606:50c0:8002::153`, `2606:50c0:8003::153`;
   `www`: `CNAME` to `ssarangi.github.io`. In Cloudflare set these records to "DNS only" (grey cloud).
3. **Settings → Pages → Custom domain**, save, wait for the certificate, then tick **Enforce HTTPS**.
