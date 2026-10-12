---
title: Project Name
page_title: Project Name: a short subtitle for the browser tab and link previews
description: One or two sentences about the project, used for search results and link previews.
tagline: What it is, in one line
author: Arpon Kapuria
author_url: https://arponkapuria.github.io
stack: Python, FastAPI, Postgres, Docker
cover: David - The Death of Socrates.jpg
cover_position: center 40%
---

<!--
HOW TO ADD A PROJECT
1. Copy this folder:  projects/_template  ->  projects/<your-slug>
2. Edit projects/<your-slug>/index.md  (this file)
3. Run:  node scripts/prerender.js     (writes projects/<your-slug>/index.html)
4. Open /projects/<your-slug>/index.html

Front matter (the block above)
  title          big heading on the page
  page_title     optional: browser-tab / link-preview title (default: "<title> | Arpon Kapuria")
  description    meta description
  tagline        line under the title
  author, author_url, stack   the two "byline" lines (all optional)
  cover          optional: Wikimedia Commons file name (or a URL) faded in over the pine drawing
  cover_position optional: CSS background-position for the cover
  image, tags    optional: link-preview image, [keyword, list]

Every "## Heading" becomes a Contents entry. Add {#short-id} to control its link.
This comment is not shown on the page.
-->

## Overview {#overview}

One paragraph on what the project is and who it is for. Plain markdown works everywhere: **bold**, *italic*, `inline code`, [links](https://example.com), lists and images.

- A bullet point
- Another one

## System diagram {#system}

A diagram is a fenced block tagged `diagram`. The header lines say what to draw around it; after the `---` line comes the spec. Keep each `id` stable, because it seeds the hand-drawn wobble.

```diagram
id: d-demo
size: 700 220
alt: A request goes from the browser to an API, which reads from a database
caption: The smallest possible diagram: three boxes and two arrows.
---
{
  nodes: [
    ['br', 'Browser', 30, 70, 150, 70, 'white'],
    ['api', 'API', 275, 70, 150, 70, 'blue'],
    ['db', 'Postgres', 520, 55, 150, 100, 'grey', 'db']
  ],
  edges: [
    ['br', 'api', 'request', 'r', 'l'],
    ['api', 'db', 'query', 'r', 'l']
  ]
}
```

## One component in depth {#component}

Short intro to the component.

#### A smaller heading

`####` gives a smaller heading. `#####` gives the bold label used for "Flow" and "Explanation".

##### Flow

1. Numbered lists render as the project's "flow" list.
2. Each step is one sentence.
3. Keep them short.

##### Explanation

A code block with a file name above it: put the language and then the path after the opening fence.

```py services/api/app/example.py
async def touch(self, corpus_id: str) -> None:
    await client.zadd(KEY, {corpus_id: time.time()}, xx=True)
```

:::qa Why was it built this way?
Collapsible Q&A. The title is the question, and the body is normal markdown, so you can use `code`, **bold**, lists and more here.
:::

:::qa What would you change next?
Another answer.
:::

:::gaps
##### Trade-offs and gaps

- The call-out box lists known limitations.
- Each bullet is one honest sentence.
:::

## Tech stack {#stack}

A normal markdown table is scrollable on small screens. Wrap it in `:::table` to add the `stack` class and column widths.

:::table stack 20% 35%
| Choice | Why I picked it | Why not the alternatives |
| --- | --- | --- |
| **FastAPI** | Async I/O fits a workload that mostly waits on the network. | **Flask**: sync-first.<br>**Django**: heavier than needed. |
:::

## What's next? {#next}

- The next thing
- And the one after

---

:::note
Source code: [github.com/arponkapuria/your-repo](https://github.com/arponkapuria/your-repo)
:::
