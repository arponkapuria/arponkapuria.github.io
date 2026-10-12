// scripts/prerender-projects.js
// Generates the static, crawler-readable HTML shell for each project write-up.
//   projects/<slug>/index.md   (you write this)   ->   projects/<slug>/index.html   (generated)
// Run: node scripts/prerender-projects.js   (also runs as part of: node scripts/prerender.js)
// Folders starting with "_" or "." (for example projects/_template) are skipped.

const fs = require("fs");
const path = require("path");

const SITE_URL = "https://arponkapuria.github.io";
const PROJECTS_DIR = path.join(__dirname, "..", "projects");
const TEMPLATE_PATH = path.join(__dirname, "prerender-project-template.html");
const COVER_SVG_PATH = path.join(__dirname, "../images/cover-pines.svg");
const DEFAULT_IMAGE = `${SITE_URL}/images/android-chrome-192x192.png`;

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

function resolveImageUrl(raw, pageUrl) {
  if (!raw) return null;
  try {
    return new URL(raw, pageUrl).href;
  } catch {
    return null;
  }
}

function extractFirstImage(body) {
  const match = body.match(/!\[[^\]]*\]\(([^)\s]+)/);
  return match ? match[1] : null;
}

function main() {
  if (!fs.existsSync(PROJECTS_DIR)) {
    console.error(`Projects dir not found: ${PROJECTS_DIR}`);
    process.exit(1);
  }

  const template = fs.readFileSync(TEMPLATE_PATH, "utf8");
  // The ink-wash cover is inlined (not an <img>) so it renders exactly like the original page.
  const coverSvg = fs.readFileSync(COVER_SVG_PATH, "utf8")
    .trim()
    .replace("<svg ", '<svg class="cover-art" aria-hidden="true" ');
  const slugs = fs.readdirSync(PROJECTS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory() && !/^[_.]/.test(d.name))
    .map(d => d.name)
    .filter(slug => fs.existsSync(path.join(PROJECTS_DIR, slug, "index.md")));

  for (const slug of slugs) {
    const raw = fs.readFileSync(path.join(PROJECTS_DIR, slug, "index.md"), "utf8");
    const { meta, body } = parseFrontmatter(raw);

    const title = meta.title || slug;
    const pageTitle = meta.page_title || `${title} | Arpon Kapuria`;
    const ogTitle = meta.page_title || title;
    const description = meta.description || meta.tagline || "";
    const pageUrl = `${SITE_URL}/projects/${slug}/`;

    const rawImage = meta.image || extractFirstImage(body);
    const resolved = resolveImageUrl(rawImage, pageUrl);
    const image = resolved || DEFAULT_IMAGE;
    const dimensionTags = resolved
      ? "" // unknown actual dimensions for a custom image: omit rather than lie to crawlers
      : `<meta property="og:image:width" content="512">\n<meta property="og:image:height" content="512">`;

    const keywords = Array.isArray(meta.tags) && meta.tags.length ? meta.tags.join(", ") : null;
    const keywordsTag = keywords ? `<meta name="keywords" content="${escapeHtml(keywords)}">` : "";

    const html = template
      .replaceAll("{{TITLE}}", escapeHtml(pageTitle))
      .replaceAll("{{OG_TITLE}}", escapeHtml(ogTitle))
      .replaceAll("{{DESCRIPTION}}", escapeHtml(description))
      .replaceAll("{{KEYWORDS_TAG}}", keywordsTag)
      .replaceAll("{{URL}}", pageUrl)
      .replaceAll("{{IMAGE}}", image)
      .replaceAll("{{IMAGE_DIMENSIONS}}", dimensionTags)
      .replaceAll("{{IMAGE_ALT}}", escapeHtml(title))
      .replaceAll("{{SLUG}}", slug)
      .replace("{{COVER_SVG}}", () => coverSvg);

    fs.writeFileSync(path.join(PROJECTS_DIR, slug, "index.html"), html);
    console.log(`✓ projects/${slug}`);
  }

  console.log(`\nGenerated ${slugs.length} project page${slugs.length === 1 ? "" : "s"} in ${PROJECTS_DIR}`);
}

module.exports = main;
if (require.main === module) main();
