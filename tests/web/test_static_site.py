"""Structure and offline-resource checks for the static ZMWiki homepage.

Uses only the standard library: no browser, no network, no JavaScript.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "src" / "zmwiki" / "web" / "static"
HTML_PATH = STATIC_DIR / "index.html"
CSS_PATH = STATIC_DIR / "styles.css"

REPO_URL = "https://github.com/lip20041219/zmwiki"

FEATURE_TITLES = (
    "Structured Wiki Pages",
    "Metadata-First Search",
    "Evidence-Aware Reading",
    "Budgeted Multi-Step Agent",
    "Source-Aware Compilation",
    "Measurable Retrieval",
)

REQUIRED_SECTION_IDS = {
    "main", "top", "capabilities", "features",
    "workflow", "terminal", "roadmap", "faq", "footer",
}

RESOURCE_TAGS = ("link", "script", "img", "source", "iframe",
                 "object", "embed", "video", "audio", "track")


class _SiteParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.doctype = None
        self.tags = []
        self.ids = set()
        self.text = []

    def handle_decl(self, decl):
        self.doctype = decl.strip().lower()

    def handle_starttag(self, tag, attrs):
        attrs = {key.lower(): (value or "") for key, value in attrs}
        self.tags.append((tag.lower(), attrs))
        if attrs.get("id"):
            self.ids.add(attrs["id"])

    def handle_data(self, data):
        self.text.append(data)


def _parse():
    parser = _SiteParser()
    parser.feed(HTML_PATH.read_text(encoding="utf-8"))
    parser.close()
    return parser


def _text(parser):
    return re.sub(r"\s+", " ", " ".join(parser.text)).strip()


def _tags(parser, name):
    return [attrs for tag, attrs in parser.tags if tag == name]


def test_document_structure_and_viewport():
    parser = _parse()
    assert parser.doctype == "doctype html"

    html = _tags(parser, "html")[0]
    assert html["lang"] == "en"

    metas = _tags(parser, "meta")
    assert any(meta.get("charset", "").lower() == "utf-8" for meta in metas)
    viewport = [meta for meta in metas if meta.get("name") == "viewport"]
    assert viewport, "missing viewport meta"
    assert "width=device-width" in viewport[0]["content"]
    assert "initial-scale=1" in viewport[0]["content"]

    title = re.search(r"<title>(.*?)</title>", HTML_PATH.read_text(encoding="utf-8"), re.S)
    assert title and "ZMWiki" in title.group(1)


def test_sections_and_navigation_targets():
    parser = _parse()
    assert REQUIRED_SECTION_IDS <= parser.ids

    nav_links = [
        attrs["href"]
        for attrs in _tags(parser, "a")
        if attrs.get("href", "").startswith("#")
    ]
    assert "#features" in nav_links
    assert "#workflow" in nav_links
    assert "#faq" in nav_links
    assert "#terminal" in nav_links
    for href in nav_links:
        assert href[1:] in parser.ids, f"anchor target missing: {href}"


def test_feature_sections_present():
    text = _text(_parse())
    for title in FEATURE_TITLES:
        assert title in text, f"missing feature: {title}"


def test_workflow_and_roadmap_content():
    parser = _parse()
    text = _text(parser)
    for step in ("Ingest", "Compile & Index", "Search", "Read", "Answer"):
        assert step in text, f"missing workflow step: {step}"
    assert "Available" in text
    assert "In Progress" in text
    assert "workflow" in parser.ids
    assert "roadmap" in parser.ids


def test_faq_uses_native_details():
    parser = _parse()
    assert "faq" in parser.ids
    assert len(_tags(parser, "details")) >= 5
    assert len(_tags(parser, "summary")) >= 5


def test_no_external_or_script_resources():
    parser = _parse()
    assert not _tags(parser, "script"), "this phase must not include JavaScript"

    stylesheets = [link for link in _tags(parser, "link")
                   if "stylesheet" in link.get("rel", "")]
    assert len(stylesheets) == 1
    assert stylesheets[0].get("href") == "styles.css"

    for tag in RESOURCE_TAGS:
        for attrs in _tags(parser, tag):
            for key in ("src", "href"):
                value = attrs.get(key, "")
                if not value or value.startswith("#") or value.startswith("data:"):
                    continue
                assert not value.startswith(("http://", "https://", "//")), (
                    f"external resource referenced by <{tag}>: {value}"
                )

    css = CSS_PATH.read_text(encoding="utf-8")
    assert "url(" not in css, "CSS must not reference external or image resources"
    assert "@import" not in css
    assert "@font-face" not in css


def test_github_links_point_to_repository():
    github_links = [
        attrs["href"]
        for attrs in _tags(_parse(), "a")
        if "github.com" in attrs.get("href", "")
    ]
    assert REPO_URL in github_links
    for href in github_links:
        assert href.startswith(REPO_URL), f"wrong GitHub repository: {href}"


def test_no_fictional_commands_or_wrong_license():
    text = _text(_parse())
    lowered = text.lower()

    for forbidden in (
        "zmwiki serve",
        "python -m zmwiki.web",
        "localhost:",
        "pip install zmwiki",
        "npm install",
        "docker run",
    ):
        assert forbidden not in lowered, f"fictional command present: {forbidden}"

    assert "pip install -e ." in text
    assert f"git clone {REPO_URL}" in text
    assert "Proprietary" in text
    assert not re.search(r"\bMIT\b", text), "license must match pyproject (Proprietary)"
    assert not re.search(r"\bApache\b", text)


def test_inline_svg_illustrations_present():
    parser = _parse()
    assert len(_tags(parser, "svg")) >= 3
    assert not _tags(parser, "image")
    assert not _tags(parser, "img")


def test_stylesheet_responsive_and_accessible():
    css = CSS_PATH.read_text(encoding="utf-8")
    for token in ("--bg:", "--text:", "--accent:", "--line:"):
        assert token in css
    assert "clamp(" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert "@media (max-width: 960px)" in css
    assert "@media (max-width: 640px)" in css
    assert "overflow-x: auto" in css
