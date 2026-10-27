from lens.nodes import Node, extract

html = '<button backend_node_id="1" aria_label="Buy">Now</button>'
node = extract(html, ["1"])[0]
assert node == Node("1", "button", "Buy Now")
assert node.line() == '[1] button "Buy Now"'
assert extract(html, ["9"]) == []
long = '<button backend_node_id="1">' + ' '.join(f'word{i}' for i in range(60)) + '</button>'
assert len(extract(long, ["1"])[0].text.split()) == 30
frame = '<html><body><iframe backend_node_id="8"><button backend_node_id="7">Go</button></iframe></body></html>'
assert extract(frame, ["7"]) == [Node("7", "button", "Go")]
print("node extraction ok")
