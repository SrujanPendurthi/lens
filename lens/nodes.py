from dataclasses import dataclass

from lxml import etree

ATTRS = ("aria_label", "placeholder", "value", "alt", "title", "name", "type")


@dataclass
class Node:
    ref: str
    tag: str
    text: str
    context: str = ""

    def line(self):
        return f'[{self.ref}] {self.tag} "{self.text}"'


def extract(cleaned_html, refs, max_words=30):
    """Return a Node for each ref (backend_node_id) found in cleaned_html, in ref order."""
    root = etree.fromstring(cleaned_html.encode(), etree.HTMLParser())
    by_id = {e.get("backend_node_id"): e for e in root.iter() if e.get("backend_node_id")}
    nodes = []
    for ref in refs:
        e = by_id.get(ref)
        if e is None:
            continue
        parts = [e.get(a) for a in ATTRS if e.get(a)] + ["".join(e.itertext())]
        text = " ".join(" ".join(parts).split()[:max_words])
        own = " ".join(" ".join(e.itertext()).split())
        context = ""
        for ancestor in e.iterancestors():
            surrounding = " ".join(" ".join(ancestor.itertext()).split())
            if len(surrounding.split()) >= 12:
                context = " ".join(surrounding.replace(own, "", 1).split()[:40]) if own else " ".join(surrounding.split()[:40])
                break
        nodes.append(Node(ref, e.tag, text, context))
    return nodes


def page_words(cleaned_html: str, refs: list[str]) -> list[tuple[str, str | None]]:
    """Visible page words in document order, owned by the innermost candidate."""
    root = etree.fromstring(cleaned_html.encode(), etree.HTMLParser())
    candidates = set(refs)
    words = []

    def emit(text, ref):
        words.extend((word, ref) for word in (text or "").split())

    def walk(element, owner=None):
        if not isinstance(element.tag, str) or element.tag in {"script", "style", "head"}:
            return
        ref = element.get("backend_node_id")
        if ref in candidates:
            owner = ref
            for attr in ("aria_label", "placeholder", "value", "alt", "title"):
                emit(element.get(attr), owner)
        emit(element.text, owner)
        for child in element:
            walk(child, owner)
            emit(child.tail, owner)

    if root is not None:
        walk(root)
    return words
