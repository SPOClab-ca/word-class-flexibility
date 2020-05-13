import unittest
import src.semantic_embedding

class WordPieceMatchTest(unittest.TestCase):
  def test_english_basic(self):
    embedder = src.semantic_embedding.SemanticEmbedding([])
    assert embedder._match_wordpiece(['I', 'have', 'a', 'cat'], 'i', 0) == [0]
    assert embedder._match_wordpiece(['I', 'have', 'a', 'cat'], 'have', 1) == [1]
    assert embedder._match_wordpiece(['I', 'have', 'a', 'cat'], 'a', 2) == [2]
    assert embedder._match_wordpiece(['I', 'have', 'a', 'cat'], 'cat', 3) == [3]
    assert embedder._match_wordpiece(['I', 'have', 'a', 'cat'], 'in', 0) == None
    assert embedder._match_wordpiece(['I', 'have', 'a', 'cat'], 'ca', 3) == None

  def test_english_split(self):
    embedder = src.semantic_embedding.SemanticEmbedding([])
    assert embedder._match_wordpiece(['what', 'a', 'coin', '##cid', '##ence', '!'], 'coincidence', 2) == [2, 3, 4]
    assert embedder._match_wordpiece(['what', 'a', 'coin', '##cid', '##ence', '!'], 'cidence', 3) == None
    assert embedder._match_wordpiece(['what', 'a', 'coin', '##cid', '##ence', '!'], 'cid', 3) == None

  def test_chinese(self):
    embedder = src.semantic_embedding.SemanticEmbedding([])
    assert embedder._match_wordpiece(['我', '是', '加', '拿', '大', '人'], '我', 0) == [0]
    assert embedder._match_wordpiece(['我', '是', '加', '拿', '大', '人'], '加拿大', 2) == [2, 3, 4]
    assert embedder._match_wordpiece(['我', '是', '加', '拿', '大', '人'], '加', 2) == [2]

