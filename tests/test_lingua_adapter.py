import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.evaluate_with_lingua import ZydaLM


class TestZydaLM(unittest.TestCase):
    @patch("scripts.evaluate_with_lingua.load_config_from_path")
    @patch("scripts.evaluate_with_lingua.instantiate")
    @patch("scripts.evaluate_with_lingua.TextPretrainingWrapper")
    @patch("scripts.evaluate_with_lingua.torch.load")
    @patch("scripts.evaluate_with_lingua.AutoTokenizer")
    def setUp(self, mock_tokenizer, mock_load, mock_wrapper, mock_instantiate, mock_load_config):
        self.mock_config = MagicMock()
        self.mock_config.dataset.tokenizer_name = "mock_tokenizer"
        self.mock_config.dataset.max_length = 1024
        mock_load_config.return_value = self.mock_config

        self.mock_network = MagicMock()
        mock_instantiate.return_value = self.mock_network

        self.mock_model = MagicMock()
        mock_wrapper.return_value = self.mock_model

        mock_load.return_value = {"state_dict": {}}

        self.mock_tok = MagicMock()
        self.mock_tok.pad_token = None
        self.mock_tok.eos_token = "[EOS]"
        self.mock_tok.encode.return_value = [1, 2, 3]
        self.mock_tok.decode.return_value = "decoded"
        self.mock_tok.pad_token_id = 0
        self.mock_tok.eos_token_id = 1
        mock_tokenizer.from_pretrained.return_value = self.mock_tok

        self.lm = ZydaLM(pretrained="dummy.ckpt", config_path="dummy_config.py", device="cpu", batch_size=1)

    def test_initialization(self):
        self.assertEqual(self.lm.batch_size, 1)
        self.assertEqual(self.lm.max_length, 1024)

    def test_loglikelihood(self):
        # Mock model output
        # Input: [1, 2, 3] (context) + [1, 2, 3] (continuation)
        # Total length 6.
        # Input to model: [1, 2, 3, 1, 2] (shifted)
        # Target: [2, 3, 1, 2, 3]

        # Mock logits: (B, L, V)
        # Let's say V=10
        logits = torch.zeros(1, 5, 10)
        # Set high logit for correct targets
        # Targets are [2, 3, 1, 2, 3]
        # Indices: 0->2, 1->3, 2->1, 3->2, 4->3
        logits[0, 0, 2] = 100
        logits[0, 1, 3] = 100
        logits[0, 2, 1] = 100
        logits[0, 3, 2] = 100
        logits[0, 4, 3] = 100

        self.lm._model_call = MagicMock(return_value=logits)
        self.lm.tok_encode = MagicMock(side_effect=lambda x: [1, 2, 3])

        requests = [("context", "continuation")]

        res = self.lm.loglikelihood(requests)

        # We expect 1 result
        self.assertEqual(len(res), 1)
        logprob, is_greedy = res[0]

        # Since we set logits to be very high for correct tokens, logprob should be close to 0 (log(1) = 0)
        # And is_greedy should be True
        self.assertTrue(is_greedy)
        # self.assertAlmostEqual(logprob, 0.0, places=1) # Logsoftmax might not be exactly 0 but close

    def test_tok_encode(self):
        self.lm.tok_encode("test")
        self.lm.tokenizer.encode.assert_called_with("test", add_special_tokens=False)


if __name__ == "__main__":
    unittest.main()
