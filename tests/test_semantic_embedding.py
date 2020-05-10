import unittest
import src.semantic_embedding

class WordPieceMatchTest(unittest.TestCase):
  def test_english_basic(self):
    embedder = src.semantic_embedding.SemanticEmbedding([])
    assert embedder._wordpiece_matches(['I', 'have', 'a', 'cat'], 'i', 0)
    assert embedder._wordpiece_matches(['I', 'have', 'a', 'cat'], 'have', 1)
    assert embedder._wordpiece_matches(['I', 'have', 'a', 'cat'], 'a', 2)
    assert embedder._wordpiece_matches(['I', 'have', 'a', 'cat'], 'cat', 3)
    assert not embedder._wordpiece_matches(['I', 'have', 'a', 'cat'], 'in', 0)
    assert not embedder._wordpiece_matches(['I', 'have', 'a', 'cat'], 'ca', 3)

  def test_english_split(self):
    embedder = src.semantic_embedding.SemanticEmbedding([])
    assert embedder._wordpiece_matches(['what', 'a', 'coin', '##cid', '##ence', '!'], 'coincidence', 2)
    assert not embedder._wordpiece_matches(['what', 'a', 'coin', '##cid', '##ence', '!'], 'cidence', 3)
    assert not embedder._wordpiece_matches(['what', 'a', 'coin', '##cid', '##ence', '!'], 'cid', 3)

  def test_chinese(self):
    embedder = src.semantic_embedding.SemanticEmbedding([])
    assert embedder._wordpiece_matches(['我', '是', '加', '拿', '大', '人'], '我', 0)
    assert embedder._wordpiece_matches(['我', '是', '加', '拿', '大', '人'], '加拿大', 2)
    assert embedder._wordpiece_matches(['我', '是', '加', '拿', '大', '人'], '加', 2)

