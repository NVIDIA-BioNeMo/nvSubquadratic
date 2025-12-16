"""
Evaluation script for Zyda models using Lingua's evaluation logic (lm-eval).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import config loader (assuming we can load python config files)
import importlib.util

from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from experiments.lightning_wrappers.text_pretraining_wrapper import TextPretrainingWrapper
from experiments.utils.cli import apply_config_overrides
from nvsubquadratic.lazy_config import instantiate


def load_config_from_path(path):
    path = Path(path).resolve()
    config_dir = path.parent
    module_name = path.stem

    # Add config dir to sys.path
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))

    try:
        module = importlib.import_module(module_name)
        return module.get_config()
    finally:
        # Optional: remove from sys.path if we want to be clean
        # sys.path.remove(str(config_dir))
        pass


@register_model("zyda")
class ZydaLM(LM):
    def __init__(
        self,
        pretrained: str,  # checkpoint path
        config_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self._device = torch.device(device)
        self.batch_size_per_gpu = batch_size

        # Load configuration
        print(f"Loading config from {config_path}")
        self.cfg = load_config_from_path(config_path)

        # Resolve interpolations
        self.cfg = apply_config_overrides(self.cfg, [])

        # Instantiate model
        print("Instantiating network...")
        self.network = instantiate(self.cfg.net)

        # Instantiate wrapper
        self.model = TextPretrainingWrapper(
            network=self.network,
            cfg=self.cfg,
            vocab_size=self.cfg.dataset.tokenizer_name
            if isinstance(self.cfg.dataset.tokenizer_name, int)
            else 50257,  # Fallback
        )

        # Load checkpoint
        print(f"Loading checkpoint from {pretrained}")
        checkpoint = torch.load(pretrained, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Handle potential prefix issues (e.g. "network." prefix if saved from wrapper but loading into wrapper)
        # The wrapper has "network" submodule.
        # If state_dict keys start with "network.", they fit directly into wrapper.
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        tokenizer_name = self.cfg.dataset.tokenizer_name
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.cfg.dataset.max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: input_ids tensor of shape (batch, seqlen)
        """
        with torch.no_grad():
            # The model expects {"input": ..., "condition": ...}
            out = self.model.network({"input": inps, "condition": None})
            return out["logits"]

    def _model_generate(self, context, max_length, eos_token_id):
        # Simple greedy generation for now
        # context: (batch, seqlen)
        pass

    def loglikelihood(self, requests):
        new_reqs = []
        for req in requests:
            # Handle both tuple (legacy) and Instance (v0.4+)
            if isinstance(req, tuple):
                context, continuation = req
            else:
                # Assuming lm_eval.api.instance.Instance
                context, continuation = req.args

            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        for chunk in tqdm(
            [requests[i : i + self.batch_size] for i in range(0, len(requests), self.batch_size)],
            disable=disable_tqdm,
            desc="Evaluating loglikelihoods",
        ):
            inputs = []
            targets = []
            ctx_lens = []

            for (context, continuation), context_enc, continuation_enc in chunk:
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                target = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][1:],
                    dtype=torch.long,
                    device=self.device,
                )
                ctx_len = len(context_enc)

                # Pad
                # We need to pad to max length in the batch
                inputs.append(inp)
                targets.append(target)
                ctx_lens.append(ctx_len)

            # Pad batch
            max_len = max(x.shape[0] for x in inputs)
            padded_inputs = []
            padded_targets = []

            for inp, tgt in zip(inputs, targets):
                pad_len = max_len - inp.shape[0]
                padded_inp = F.pad(inp, (0, pad_len), value=self.tokenizer.pad_token_id)
                padded_tgt = F.pad(tgt, (0, pad_len), value=-100)  # -100 for ignore index

                # Mask for attention (if needed by model, but here we just pass input_ids)
                # Assuming model handles padding via attention mask if passed,
                # but TextPretrainingWrapper/Network might expect explicit mask.
                # The wrapper's forward doesn't take mask in the dict, but network might.
                # Zyda network usually takes input_ids.
                # Let's assume right padding for now and model handles it or we pass mask?
                # The wrapper forward: self.network({"input": input_ids, "condition": None})
                # We should check if network supports padding.
                # For now, let's just pass padded inputs.

                padded_inputs.append(padded_inp)
                padded_targets.append(padded_tgt)

            input_batch = torch.stack(padded_inputs)
            _ = torch.stack(padded_targets)

            logits = self._model_call(input_batch)

            # Compute logprobs
            log_softmax = F.log_softmax(logits, dim=-1)

            # Gather logprobs of targets
            # target_batch: (B, L)
            # log_softmax: (B, L, V)

            for i, (inp, tgt, ctx_len) in enumerate(zip(inputs, targets, ctx_lens)):
                # Slice out the relevant parts (remove padding)
                # The original inp and tgt length
                seq_len = inp.shape[0]

                # Logprobs for this sequence
                seq_logprobs = log_softmax[i, :seq_len, :]

                # Gather target logprobs
                # tgt is (L,)
                # We want logprobs at indices tgt

                # Continuation starts at ctx_len - 1 in the input (because input is shifted by 1 relative to original seq)
                # Actually:
                # Original: [ctx..., cont...]
                # Input:    [ctx..., cont...][:-1]
                # Target:   [ctx..., cont...][1:]
                # We want likelihood of continuation tokens.
                # Continuation tokens in Target start at index `ctx_len - 1`?
                # Example: ctx=[A], cont=[B]
                # Full=[A, B]
                # Input=[A]
                # Target=[B]
                # ctx_len=1
                # We want logprob of B given A.
                # This is at index 0 of output (corresponding to input A).
                # So we want range [ctx_len-1 : ]

                # Wait, if ctx is empty? handled above.

                relevant_tgt = tgt[ctx_len - 1 :]
                relevant_logprobs = seq_logprobs[ctx_len - 1 :]

                # Gather
                greedy_tokens = relevant_logprobs.argmax(dim=-1)
                max_equal = (greedy_tokens == relevant_tgt).all()

                target_logprobs = torch.gather(relevant_logprobs, -1, relevant_tgt.unsqueeze(-1)).squeeze(-1)

                res.append((target_logprobs.sum().item(), max_equal.item()))

        return res

    def loglikelihood_rolling(self, requests):
        # TODO: Implement rolling loglikelihood for perplexity tasks
        raise NotImplementedError("loglikelihood_rolling not implemented")

    def generate_until(self, requests):
        # Basic generation implementation
        # For now, raise NotImplementedError or implement simple greedy
        raise NotImplementedError("Generation not yet implemented for ZydaLM")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--tasks", type=str, default="hellaswag", help="Comma separated tasks")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()

    lm = ZydaLM(
        pretrained=args.ckpt_path, config_path=args.config_path, device=args.device, batch_size=args.batch_size
    )

    tasks = args.tasks.split(",")

    results = simple_evaluate(model=lm, tasks=tasks, batch_size=args.batch_size, device=args.device)

    print(results["results"])

    if args.output_path:
        import json

        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
