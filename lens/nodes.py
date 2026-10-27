from dataclasses import dataclass
import re

from lxml import etree

ATTRS = ("aria_label", "placeholder", "value", "alt", "title", "name", "type")


@dataclass
class Node:
    ref: str
    tag: str
    text: str

    def line(self):
        return f'[{self.ref}] {self.tag} "{self.text}"'


def _page_root(cleaned_html):
    # Dataset iframe contents are expanded DOMs, not HTML's raw-text fallback.
    expanded = re.sub(r"(</?)iframe(?=[\s>])", r"\1lens-iframe", cleaned_html, flags=re.I)
    root = etree.fromstring(expanded.encode(), etree.HTMLParser())
    if root is not None:
        for element in root.iter("lens-iframe"):
            element.tag = "iframe"
    return root


def extract(cleaned_html, refs, max_words=30):
    """Return a Node for each ref (backend_node_id) found in cleaned_html, in ref order."""
    root = _page_root(cleaned_html)
    by_id = {e.get("backend_node_id"): e for e in root.iter() if e.get("backend_node_id")}
    nodes = []
    for ref in refs:
        e = by_id.get(ref)
        if e is None:
            continue
        parts = [e.get(a) for a in ATTRS if e.get(a)] + ["".join(e.itertext())]
        text = " ".join(" ".join(parts).split()[:max_words])
        nodes.append(Node(ref, e.tag, text))
    return nodes
