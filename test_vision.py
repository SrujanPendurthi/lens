import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from lens import vision


class GeometryTests(unittest.TestCase):
    def test_tile_tops(self):
        self.assertEqual(vision.tile_tops(1889), [0, 1280])
        self.assertEqual(vision.tile_tops(1280), [0])
        self.assertEqual(vision.tile_tops(0), [])

    def test_box_patches_basic(self):
        self.assertEqual(vision.box_patches([0, 0, 40, 40], 0), [0])
        self.assertEqual(vision.box_patches([534, 45, 151, 22], 0), [45, 46, 47, 48, 49])

    def test_box_spanning_tiles(self):
        box = [0, 1270, 40, 20]
        self.assertEqual(vision.box_patches(box, 0), [31 * 32])
        self.assertEqual(vision.box_patches(box, 1280), [0])

    def test_box_edge_cases(self):
        for box in ([0, 0, 0, 0], [0, 5000, 40, 40], [0, 0, -1, 40], None,
                    [float('nan'), 0, 40, 40], [1, 2]):
            self.assertEqual(vision.box_patches(box, 0), [])
        self.assertEqual(vision.box_patches([-20, -20, 40, 40], 0), [0])
        self.assertEqual(vision.box_patches([1270, 0, 40, 40], 0), [31])

    def test_loads_jpeg_named_png(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)):
            Image.new('RGB', (1280, 1500), 'black').save(Path(d) / 'abc.png', format='JPEG')
            tiles = vision.load_tiles('abc')
            self.assertEqual(len(tiles), 2)
            self.assertTrue(all(tile.size == (1280, 1280) and tile.mode == 'RGB' for tile in tiles))
            self.assertEqual(tiles[1].getpixel((0, 219)), (0, 0, 0))
            self.assertEqual(tiles[1].getpixel((0, 220)), (255, 255, 255))

    def test_extra_right_margin_is_clipped_without_rescaling(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)):
            image = Image.new('RGB', (1410, 1080), 'black')
            image.putpixel((1279, 0), (255, 0, 0))
            image.save(Path(d) / 'wide.png')
            tiles = vision.load_tiles('wide')
            self.assertEqual(tiles[0].size, (1280, 1280))
            self.assertEqual(tiles[0].getpixel((1279, 0)), (255, 0, 0))
            self.assertEqual(tiles[0].getpixel((0, 1080)), (255, 255, 255))

    def test_missing_screenshot(self):
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)), \
             patch.object(vision, 'CACHE_DIR', Path(d) / 'cache'), \
             patch.object(vision, '_load', side_effect=AssertionError('must not load model')):
            self.assertIsNone(vision.load_tiles('x'))
            self.assertIsNone(vision.encode_tiles('x'))
            scores = vision.rank_colpali('q', 'x', ['1'], {'1': [0, 0, 10, 10]})
            self.assertEqual(scores[0].item(), float('-inf'))
            self.assertTrue(vision.encoding_stats('x')['no_screenshot'])

    def test_image_token_selection(self):
        ids = torch.tensor([[11, 99, 99, 22, 99]])
        embeddings = torch.arange(10).reshape(1, 5, 2).float()
        selected = vision.image_embeddings(embeddings, ids, 99)
        self.assertTrue(torch.equal(selected, embeddings[0, [1, 2, 4]]))

    def test_rank_clips_bottom_and_pools_across_tiles(self):
        from PIL import Image
        tiles = [torch.zeros((1024, 2)), torch.zeros((1024, 2))]
        tiles[0][31 * 32] = torch.tensor([1., 0.])
        tiles[1][0] = torch.tensor([0., 1.])
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)), \
             patch.object(vision, 'encode_tiles', return_value=tiles), \
             patch.object(vision, 'encode_query', return_value=torch.eye(2)):
            Image.new('RGB', (1280, 1290), 'white').save(Path(d) / 'abc.png')
            scores = vision.rank_colpali('q', 'abc', ['span', 'below', 'partial', 'missing', 'zero'],
                                         {'span': [0, 1270, 40, 20], 'below': [0, 1290, 40, 10],
                                          'partial': [0, 1285, 40, 20], 'zero': [0, 0, 0, 0]})
            self.assertEqual(scores.tolist(), [1., float('-inf'), 1., float('-inf'), float('-inf')])

    def test_nested_child_beats_container(self):
        from PIL import Image
        tile = torch.zeros((1024, 2))
        tile[0] = torch.tensor([1., 0.])
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)), \
             patch.object(vision, 'encode_tiles', return_value=[tile]), \
             patch.object(vision, 'encode_query', return_value=torch.eye(2)[:1]):
            Image.new('RGB', (1280, 1280), 'white').save(Path(d) / 'nested.png')
            scores = vision.rank_colpali('q', 'nested', ['child', 'container'],
                                         {'child': [0, 0, 40, 40], 'container': [0, 0, 1280, 1280]})
            self.assertGreater(scores[0], scores[1])
            self.assertEqual(scores[0].item(), 1.0)

    def test_legacy_cache_hits(self):
        tile = torch.ones((1024, 2))
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)), \
             patch.object(vision, 'CACHE_DIR', Path(d)), \
             patch.object(vision, 'load_tiles', side_effect=AssertionError('legacy cache missed')):
            (Path(d) / 'u.png').touch()
            torch.save({'tiles': [tile], 'stats': {'n_tiles': 1, 'encode_s': 2.}}, Path(d) / 'u.pt')
            loaded = vision.encode_tiles('u')
            self.assertTrue(torch.equal(loaded[0], tile))
            self.assertTrue(vision.encoding_stats('u')['cache_hit'])

    def test_stale_cache_reencodes(self):
        for metadata in ({'model': 'other'}, {'tile': 640}):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as d, \
                 patch.object(vision, 'SHOTS', Path(d)), patch.object(vision, 'CACHE_DIR', Path(d)), \
                 patch.object(vision, 'load_tiles', return_value=[]) as load, \
                 patch.object(vision, '_load', return_value=(object(), object())):
                (Path(d) / 'u.png').touch()
                torch.save({'tiles': [], 'stats': {}} | metadata, Path(d) / 'u.pt')
                vision.encode_tiles('u')
                load.assert_called_once_with('u')
                payload = torch.load(Path(d) / 'u.pt', weights_only=True)
                self.assertEqual(payload['model'], vision.MODEL)
                self.assertEqual(payload['tile'], vision.TILE)


@unittest.skipUnless(os.environ.get('LENS_VISION_TESTS'), 'loads 3B model')
class ColPaliModelTests(unittest.TestCase):
    def test_image_token_selection(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)), \
             patch.object(vision, 'CACHE_DIR', Path(d) / 'cache'):
            Image.new('RGB', (1280, 1280), 'white').save(Path(d) / 'white.png')
            embeddings = vision.encode_tiles('white')
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(embeddings[0].shape[0], 1024)
            self.assertEqual(embeddings[0].dtype, torch.float32)
            self.assertEqual(embeddings[0].device.type, 'cpu')
            self.assertTrue(torch.allclose(embeddings[0].norm(dim=1), torch.ones(1024), atol=1e-5))
            with patch.object(vision, 'load_tiles', side_effect=AssertionError('cache missed')):
                cached = vision.encode_tiles('white')
            self.assertTrue(torch.equal(cached[0], embeddings[0]))

    def test_text_region_wins(self):
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 88)
        image = Image.new('RGB', (1280, 1280), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((40, 35), 'SEARCH', fill='black', font=font)
        draw.text((40, 835), 'PRIVACY POLICY', fill='black', font=font)
        with tempfile.TemporaryDirectory() as d, patch.object(vision, 'SHOTS', Path(d)), \
             patch.object(vision, 'CACHE_DIR', Path(d) / 'cache'):
            image.save(Path(d) / 'regions.png')
            scores = vision.rank_colpali('click the search button', 'regions', ['search', 'privacy'],
                                         {'search': [0, 0, 1280, 160], 'privacy': [0, 800, 1280, 160]})
            print('real ColPali region scores:', scores.tolist(), flush=True)
            self.assertGreater(scores[0], scores[1])


if __name__ == '__main__':
    unittest.main()
