/* ============================================================
   blog.js — renders a single markdown post into article.html
   Usage: article.html?post=slug-of-file   (file = writings/slug-of-file.md)
   ============================================================ */

(function () {
  "use strict";

  const WRITINGS_DIR = "../writings/";

  /* ---------------- Theme ---------------- */
  const root = document.documentElement;
  const toggleBtn = document.getElementById("theme-toggle");

  function applyTheme(theme) {
    if (theme === "light" || theme === "dark") {
      root.setAttribute("data-theme", theme);
    } else {
      root.removeAttribute("data-theme");
    }

    const isDark = (theme === "dark") ||
    (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);

    const hlTheme = document.getElementById("hljs-theme");
    if (hlTheme) {
      hlTheme.href = isDark
        ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"
        : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css";
    }
  }

  const savedTheme = localStorage.getItem("blog-theme");
  applyTheme(savedTheme);

  toggleBtn.addEventListener("click", () => {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const current = root.getAttribute("data-theme") || (prefersDark ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem("blog-theme", next);
  });

  /* ---------------- Frontmatter parsing ----------------
     ---
     title: My Post
     description: Something
     date: 2026-01-15
     modified: 2026-01-20
     author: Arpon Kapuria
     category: AI Engineering
     tags: [rag, llm, mlops]
     ---
  -------------------------------------------------------- */
  function parseFrontmatter(raw) {
    const fmMatch = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/);
    if (!fmMatch) return { meta: {}, body: raw };

    const meta = {};
    const lines = fmMatch[1].split("\n");
    for (const line of lines) {
      const idx = line.indexOf(":");
      if (idx === -1) continue;
      const key = line.slice(0, idx).trim();
      let val = line.slice(idx + 1).trim();

      if (/^\[.*\]$/.test(val)) {
        val = val
          .slice(1, -1)
          .split(",")
          .map((s) => s.trim().replace(/^["']|["']$/g, ""))
          .filter(Boolean);
      } else {
        val = val.replace(/^["']|["']$/g, "");
      }
      meta[key] = val;
    }
    const body = raw.slice(fmMatch[0].length);
    return { meta, body };
  }

  /* ---------------- Helpers ---------------- */
  function slugify(text) {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, "")
      .replace(/\s+/g, "-");
  }

  function formatDate(dateStr) {
    if (!dateStr) return "";
    const d = new Date(dateStr);
    if (isNaN(d)) return dateStr;
    return d.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  }

  function readingTime(text) {
    const words = text.trim().split(/\s+/).filter(Boolean).length;
    const minutes = Math.max(1, Math.round(words / 100));
    return `⏱️\u00A0\u00A0${minutes} min read`;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  /* ---------------- Math protection (must run before marked) ----------------
     marked.js will corrupt LaTeX (underscores → <em>, etc.) if we let it see
     raw $ delimiters. So we extract all math spans first, replace with opaque
     tokens, parse markdown, then render with KaTeX and splice back in.
  --------------------------------------------------------------------------- */
  function protectMath(src) {
    const stash = [];
    // Display math $$...$$ first (greedy would eat inline, so do display first)
    src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => {
      stash.push({ display: true, content: m });
      return `\x02MATH${stash.length - 1}\x03`;
    });
    // Inline math $...$  (single line only, no nested $)
    src = src.replace(/\$([^\$\n]+?)\$/g, (_, m) => {
      stash.push({ display: false, content: m });
      return `\x02MATH${stash.length - 1}\x03`;
    });
    return { src, stash };
  }

  function restoreMath(html, stash) {
    return html.replace(/\x02MATH(\d+)\x03/g, (_, i) => {
      const { display, content } = stash[parseInt(i, 10)];
      try {
        return katex.renderToString(content, { displayMode: display, throwOnError: false });
      } catch (e) {
        return escapeHtml(content);
      }
    });
  }

  function processImageSizes(src) {
    // ![alt|400](url) or ![alt|400x300](url)
    return src.replace(
      /!\[([^\]]*?)\|(\d+)(?:x(\d+))?\]\(([^)]+)\)/g,
      (_, alt, w, h, url) => {
        const size = h ? `width="${w}" height="${h}"` : `width="${w}"`;
        return `<img src="${url}" alt="${alt}" ${size} style="max-width:100%" loading="lazy">`;
      }
    );
  }

  /* ---------------- Markdown renderer setup ---------------- */
  const headingRegistry = []; // {level, text, id}

  const renderer = new marked.Renderer();

  renderer.heading = function (text, level, raw) {
    if (level === 2 || level === 3) {
      let id = slugify(raw);
      let unique = id, n = 2;
      while (headingRegistry.some((h) => h.id === unique)) unique = `${id}-${n++}`;
      id = unique;
      headingRegistry.push({ level, text: raw, id });
      return `<h${level} id="${id}">${text}</h${level}>\n`;
    }
    return `<h${level}>${text}</h${level}>\n`;
  };

  renderer.link = function (href, title, text) {
    return `<a href="${href}"${title ? ` title="${escapeHtml(title)}"` : ''} target="_blank" rel="noopener noreferrer">${text}</a>`;
  };

  renderer.code = function (code, infostring) {
    const raw = (infostring || "").trim().split(/\s+/)[0];

    if (raw === "mermaid") {
      return `<div class="mermaid">${escapeHtml(code)}</div>\n`;
    }

    const aliasMap = {
      sh: "bash", shell: "bash", zsh: "bash", console: "bash",
      cuda: "cpp", cu: "cpp",
    };
    const normalized = aliasMap[raw] || raw;

    let highlighted;
    try {
      highlighted = normalized && hljs.getLanguage(normalized)
        ? hljs.highlight(code, { language: normalized }).value
        : hljs.highlightAuto(code).value;
    } catch (e) {
      highlighted = escapeHtml(code);
    }

    const langLabel = raw || "text";
    const langClass = normalized ? ` class="language-${normalized}"` : "";
    return `<div class="code-block">
      <div class="code-header">
        <span class="code-lang">${escapeHtml(langLabel)}</span>
        <button class="copy-btn" aria-label="Copy code">Copy</button>
      </div>
      <pre><code${langClass}>${highlighted}</code></pre>
    </div>\n`;
  };

  marked.setOptions({
    renderer,
    gfm: true,
    breaks: false,
    html: true,
  });

  /* ---------------- Build TOC ---------------- */
  function buildTOC() {
    const list = document.getElementById("toc-list");
    const toc = document.getElementById("toc");
    list.innerHTML = "";

    if (headingRegistry.length === 0) {
      toc.style.display = "none";
      document.querySelector(".toc-spacer")?.style.setProperty("display", "none");
      return;
    }

    for (const h of headingRegistry) {
      const li = document.createElement("li");
      if (h.level === 3) li.className = "toc-h3";
      const a = document.createElement("a");
      a.href = `#${h.id}`;
      a.textContent = h.text;
      li.appendChild(a);
      list.appendChild(li);
    }

    // Scrollspy
    const links = list.querySelectorAll("a");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const id = entry.target.id;
          const link = list.querySelector(`a[href="#${id}"]`);
          if (!link) return;
          if (entry.isIntersecting) {
            links.forEach((l) => l.classList.remove("active"));
            link.classList.add("active");
          }
        });
      },
      { rootMargin: "0px 0px -70% 0px", threshold: 1.0 }
    );
    headingRegistry.forEach((h) => {
      const el = document.getElementById(h.id);
      if (el) observer.observe(el);
    });
  }

  /* ---------------- Metadata injection ---------------- */
  function setMeta(meta, descriptionText) {
    const pageTitle = `${meta.title || "Untitled"} | Arpon Kapuria`;
    document.title = pageTitle;

    const url = window.location.href;
    const image = meta.image ? meta.image : "../images/og-default.png";

    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) {
        if (el.tagName === "LINK") el.href = val;
        else el.setAttribute("content", val);
      }
    };

    set("meta-description", descriptionText);
    set("og-title", meta.title || pageTitle);
    set("og-description", descriptionText);
    set("og-url", url);
    set("og-image", image);
    set("twitter-title", meta.title || pageTitle);
    set("twitter-description", descriptionText);
    set("twitter-image", image);
    set("meta-published", meta.date || "");
    set("meta-modified", meta.modified || meta.date || "");
    set("meta-category", meta.category || "");
    set("meta-canonical", url);
  }

  /* ---------------- Back to Top ---------------- */
  function initBackToTop() {
    const button = document.getElementById("back-to-top");

    if (!button) return;

    window.addEventListener("scroll", () => {
      button.classList.toggle("show", window.scrollY > 300);
    });

    button.addEventListener("click", () => {
      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    });
  }

  /* ---------------- Main ---------------- */
  async function init() {
    const params = new URLSearchParams(window.location.search);
    const slug = params.get("post");
    const root = document.getElementById("article-root");
    const loading = document.getElementById("loading");

    if (!slug) {
      loading.textContent = "No post specified.";
      return;
    }

    let raw;
    try {
      const res = await fetch(`${WRITINGS_DIR}${slug}.md`);
      if (!res.ok) throw new Error(`${res.status}`);
      raw = await res.text();
    } catch (err) {
      loading.textContent = "Couldn't load this post. It may have been moved or renamed.";
      return;
    }

    const { meta, body } = parseFrontmatter(raw);
    const description = meta.description || "";

    setMeta(meta, description);

    // Protect math before marked sees it, then restore after
    const { src: safeSrc, stash } = protectMath(body);
    const sized = processImageSizes(safeSrc);
    const rawHtml = marked.parse(sized);
    const html = restoreMath(rawHtml, stash);

    document.getElementById("article-title").textContent = meta.title || "Untitled";
    document.getElementById("article-description").textContent = description;

    const dateEl = document.getElementById("article-date");
    const dateText = formatDate(meta.date);
    const modText = meta.modified && meta.modified !== meta.date ? ` (Updated ${formatDate(meta.modified)})` : "";
    dateEl.textContent = dateText + modText;

    document.getElementById("article-readtime").textContent = readingTime(body);

    const contentEl = document.getElementById("article-content");
    contentEl.innerHTML = html;
    document.getElementById("article-header").hidden = false;
    loading.remove();

    if (window.mermaid) {
      mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        look: "handDrawn",
        themeVariables: {
          fontFamily: "'Space Grotesk', Inter, sans-serif",
          fontSize: "16px",
          background: "#fffdf7",
          primaryColor: "#ffd43b",
          primaryTextColor: "#1c1c1e",
          primaryBorderColor: "#1c1c1e",
          lineColor: "#1c1c1e",
          secondaryColor: "#74c0fc",
          tertiaryColor: "#ff8787",
          edgeLabelBackground: "#fffdf7",
          clusterBkg: "#f1f3f5",
          clusterBorder: "#1c1c1e",
        }
      });
      mermaid.run({ querySelector: ".mermaid" });
    }

    // Copy button handler (delegated)
    contentEl.addEventListener("click", (e) => {
      if (!e.target.matches(".copy-btn")) return;
      const block = e.target.closest(".code-block");
      const code = block?.querySelector("code")?.textContent || "";
      navigator.clipboard.writeText(code).then(() => {
        e.target.textContent = "Copied!";
        setTimeout(() => { e.target.textContent = "Copy"; }, 1800);
      }).catch(() => {
        e.target.textContent = "Failed";
        setTimeout(() => { e.target.textContent = "Copy"; }, 1800);
      });
    });

    buildTOC();
    initBackToTop();
  }

  init();
})();