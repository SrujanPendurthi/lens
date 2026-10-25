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
