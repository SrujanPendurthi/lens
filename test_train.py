import unittest
from unittest.mock import patch

import train


class TripletTests(unittest.TestCase):
    def test_triplets_skip_and_pick(self):
        steps = [
            {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"},
             {"ref": "2", "tag": "a", "text": "Stop"}, {"ref": "3", "tag": "a", "text": ""}]},
            {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": ""},
             {"ref": "2", "tag": "a", "text": "Stop"}]},
            {"task": "t", "prev": [], "gold": ["1"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"}]},
            {"task": "t", "prev": [], "gold": ["9"], "nodes": [{"ref": "1", "tag": "a", "text": "Go"}]}]
        with patch("lens.pick.subgoal", return_value={"text": "click go"}) as subgoal:
            rows, skips = train.triplets(steps)
        self.assertEqual(rows, [{"query": "click go", "positive": "Go", "negative": "Stop"}])
        self.assertEqual(skips, {"textless_gold": 1, "no_negative": 1, "no_gold": 1})
        subgoal.assert_called_once_with("t", [])


if __name__ == '__main__':
    unittest.main()
