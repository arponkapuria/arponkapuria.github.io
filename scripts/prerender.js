// scripts/prerender.js
// Generates static, crawler-readable HTML shells for each blog post.
// Run: node scripts/prerender.js

const fs = require("fs");
const path = require("path");

const SITE_URL = "https://arponkapuria.github.io";
const WRITINGS_DIR = path.join(__dirname, "..", "writings");
const OUTPUT_DIR = path.join(__dirname, "..", "blogs", "posts");
const TEMPLATE_PATH = path.join(__dirname, "prerender-template.html");
const DEFAULT_IMAGE = `${SITE_URL}/images/android-chrome-192x192.png`;
const IMAGE_BASE = `${SITE_URL}/blogs/index.html`;

function resolveImageUrl(raw) {
  if (!raw) return null;
  try {
    return new URL(raw, IMAGE_BASE).href;
  } catch {
    return null;
  }
}

function extractFirstImage(body) {
  const match = body.match(/!\[[^\]]*\]\(([^)]+)\)/);
  return match ? match[1] : null;
}

function parseFrontmatter(raw) {
  const match = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/);
  if (!match) return { meta: {}, body: raw };

  const meta = {};
  for (const line of match[1].split("\n")) {
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    let val = line.slice(idx + 1).trim();
    if (/^\[.*\]$/.test(val)) {
      val = val.slice(1, -1).split(",").map(s => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
    } else {
      val = val.replace(/^["']|["']$/g, "");
    }
    meta[key] = val;
  }
  return { meta, body: raw.slice(match[0].length) };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function slugify(text) {
  return text.toLowerCase().trim().replace(/[^\w\s-]/g, "").replace(/\s+/g, "-");
}

function main() {
  if (!fs.existsSync(WRITINGS_DIR)) {
    console.error(`Writings dir not found: ${WRITINGS_DIR}`);
    process.exit(1);
  }

  const template = fs.readFileSync(TEMPLATE_PATH, "utf8");
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const files = fs.readdirSync(WRITINGS_DIR).filter(f => f.endsWith(".md"));

  for (const file of files) {
    const slug = file.replace(/\.md$/, "");
    const raw = fs.readFileSync(path.join(WRITINGS_DIR, file), "utf8");
    const { meta, body } = parseFrontmatter(raw);

    const title = meta.title || slug;
    const description = meta.description || "";
    const pageTitle = `${title} | Arpon Kapuria`;
    const postUrl = `${SITE_URL}/blogs/posts/${slug}/`;

    const rawImage = meta.image || extractFirstImage(body);
    const image = resolveImageUrl(rawImage) || DEFAULT_IMAGE;
    const usingDefault = !resolveImageUrl(rawImage);
    const dimensionTags = usingDefault
      ? `<meta property="og:image:width" content="512">\n<meta property="og:image:height" content="512">`
      : ""; // unknown actual dimensions for article images — omit rather than lie to crawlers

    // Only emit a keywords tag when real tags exist — no invented generic fallback
    const keywords = Array.isArray(meta.tags) && meta.tags.length
      ? meta.tags.join(", ")
      : null;
    const keywordsTag = keywords
      ? `<meta name="keywords" content="${escapeHtml(keywords)}">`
      : "";

    const published = meta.date || "";
    const modified = meta.modified || meta.date || "";
    const category = meta.category || "";

    const html = template
      .replaceAll("{{TITLE}}", escapeHtml(pageTitle))
      .replaceAll("{{OG_TITLE}}", escapeHtml(title))
      .replaceAll("{{DESCRIPTION}}", escapeHtml(description))
      .replaceAll("{{KEYWORDS_TAG}}", keywordsTag)
      .replaceAll("{{URL}}", postUrl)
      .replaceAll("{{IMAGE}}", image)
      .replaceAll("{{IMAGE_DIMENSIONS}}", dimensionTags)
      .replaceAll("{{IMAGE_ALT}}", escapeHtml(title))
      .replaceAll("{{PUBLISHED}}", published)
      .replaceAll("{{MODIFIED}}", modified)
      .replaceAll("{{CATEGORY}}", escapeHtml(category))
      .replaceAll("{{SLUG}}", slug);

    const outDir = path.join(OUTPUT_DIR, slug);
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, "index.html"), html);
    console.log(`✓ ${slug}`);
  }

  console.log(`\nGenerated ${files.length} post pages in ${OUTPUT_DIR}`);
}

main();