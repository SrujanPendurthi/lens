from lens.nodes import Node, extract

html = '''<html><body>outer-only words
<section>Diamond earrings in gold with sparkling stones for a special occasion.
<button backend_node_id="1">Buy Now</button><span>Buy Now</span></section>
</body></html>'''
node = extract(html, ["1"])[0]
assert node.context == "Diamond earrings in gold with sparkling stones for a special occasion. Buy Now", node.context
assert node.line() == Node("1", "button", "Buy Now").line()
assert "outer-only" not in node.context  # smallest qualifying ancestor
long = '<div>' + ' '.join(f'word{i}' for i in range(60)) + '<button backend_node_id="1">Buy Now</button></div>'
assert len(extract(long, ["1"])[0].context.split()) == 40
print("context extraction ok")

from lens.nodes import page_words
html = '<html><body>Flights <div backend_node_id="1">noon <a backend_node_id="2"><span>View details</span></a></div> after</body></html>'
w = page_words(html, ["1", "2"])
assert w == [("Flights", None), ("noon", "1"), ("View", "2"), ("details", "2"), ("after", None)], w
w2 = page_words('<html><body><button backend_node_id="3" aria_label="Close"/></body></html>', ["3"])
assert w2 == [("Close", "3")], w2

# Mind2Web expands iframe DOMs; the HTML parser otherwise treats them as raw text.
frame = '<html><body><iframe backend_node_id="8"><button backend_node_id="7">Go</button></iframe></body></html>'
assert [n.ref for n in extract(frame, ["7"])] == ["7"]
assert page_words(frame, ["7"]) == [("Go", "7")]
