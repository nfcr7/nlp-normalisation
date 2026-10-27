# inference.py - independent inference: load artifacts and expose normalize_text(text)
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so that model_def is importable

import torch
from tokenizers import Tokenizer
from model_def import Seq2Seq, simple_tokenize, detokenize, is_bypass, make_word_inputs

MAX_TOKENS = 200   # very long messages are truncated to this many tokens

def load_normalizer(artifact_dir, device=None):
    artifact_dir = Path(artifact_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
    model = Seq2Seq(cfg["vocab_size"], cfg["emb_dim"], cfg["hidden_dim"], cfg["dropout"], cfg["pad_id"]).to(device)
    model.load_state_dict(torch.load(artifact_dir / "best_model.pt", map_location=device))
    model.eval()
    pad_id, bos_id, eos_id, sep_id = cfg["pad_id"], cfg["bos_id"], cfg["eos_id"], cfg["sep_id"]
    max_len, context = cfg["max_len"], cfg["context"]
    cache = {}

    def encode_word(word):
        if word not in cache:
            cache[word] = tokenizer.encode(word).ids
        return cache[word]

    def run_model(inputs, batch_size):
        # Greedy decoding of every input; sorted by length for speed, results returned in original order
        outputs = [""] * len(inputs)
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            ids_list = [inputs[i] for i in idx]
            lens = torch.tensor([len(x) for x in ids_list])
            src = torch.full((len(ids_list), int(lens.max())), pad_id, dtype=torch.long)
            for row, ids in enumerate(ids_list):
                src[row, :len(ids)] = torch.tensor(ids)
            limit = min(max_len, int(lens.max()) * 2 + 10)           # prevents endless generation
            generated = model.generate(src.to(device), lens.to(device), bos_id, eos_id, limit)
            for i, ids in zip(idx, generated):
                outputs[i] = tokenizer.decode(ids).strip()
        return outputs

    def normalize_tokens_batch(token_lists, batch_size=512):
        # token_lists: list of messages, each a list of tokens. Returns one output string per token.
        inputs, owners = [], []
        for m, tokens in enumerate(token_lists):
            context_tokens = ["[user]" if t.startswith("@") else t for t in tokens]
            word_inputs = make_word_inputs(context_tokens, encode_word, sep_id, eos_id, context, max_len)
            for i, ids in enumerate(word_inputs):
                if not is_bypass(tokens[i]):
                    inputs.append(ids); owners.append((m, i))
        predictions = run_model(inputs, batch_size)
        outputs = [list(tokens) for tokens in token_lists]            # mentions/URLs keep themselves
        for (m, i), prediction in zip(owners, predictions):
            outputs[m][i] = prediction
        return outputs

    def normalize_batch(texts, pretokenized=False, detok=True, batch_size=512):
        token_lists = []
        for text in texts:
            text = " ".join(str(text).split())
            tokens = text.split() if pretokenized else simple_tokenize(text)
            token_lists.append(tokens[:MAX_TOKENS])
        results = []
        for out in normalize_tokens_batch(token_lists, batch_size):
            joined = " ".join(w for w in out if w)
            results.append(detokenize(joined) if detok else joined)    # empty input -> ""
        return results

    def normalize_text(text, detok=True):
        return normalize_batch([text], detok=detok)[0]

    normalize_text.batch = normalize_batch
    normalize_text.tokens = normalize_tokens_batch
    return normalize_text
