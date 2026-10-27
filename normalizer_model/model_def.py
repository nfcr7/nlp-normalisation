# model_def.py - Seq2Seq (BiGRU encoder + attention + GRU decoder) and text helpers
import re
import torch
import torch.nn as nn

# ---------- Light tokenization / detokenization (same style as the dataset) ----------
TOKEN_REGEX = re.compile(r"https?://\S+|[@#]\w+|[:;=][-o^]?[)(DPp/\\|]|\w+(?:['’]\w+)*|[^\w\s]")

def simple_tokenize(text):
    # Split raw text into tokens: keeps mentions, hashtags, URLs and basic emoticons whole
    return TOKEN_REGEX.findall(text)

def detokenize(text):
    # Cosmetic spacing fix for display only (no space before , . ! ? ; : % ) ])
    text = re.sub(r"\s+([,.!?;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()

def is_bypass(token):
    # Mentions and URLs are random strings: they are copied, never sent to the model
    return token == "[user]" or token.startswith(("@", "http://", "https://", "www."))

def make_word_inputs(tokens, encode_word, sep_id, eos_id, context, max_len):
    # One model input per token: left context + <sep> + token + <sep> + right context + <eos>
    inputs = []
    for i, token in enumerate(tokens):
        left = [x for t in tokens[max(0, i - context):i] for x in encode_word(t)][-24:]
        right = [x for t in tokens[i + 1:i + 1 + context] for x in encode_word(t)][:24]
        inputs.append((left + [sep_id] + encode_word(token)[:40] + [sep_id] + right)[:max_len - 1] + [eos_id])
    return inputs

# ---------- Model ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))                      # (B, S, E)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h = self.gru(packed)                             # h: (2, B, H)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
        h = torch.tanh(self.fc(torch.cat([h[0], h[1]], dim=1)))      # (B, H) initial decoder state
        return out, h                                                # out: (B, S, 2H)

class Attention(nn.Module):
    # Additive (Bahdanau) attention
    def __init__(self, hidden_dim):
        super().__init__()
        self.w_enc = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.w_dec = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_out, mask):
        energy = self.v(torch.tanh(self.w_enc(enc_out) + self.w_dec(dec_hidden).unsqueeze(1))).squeeze(2)  # (B, S)
        energy = energy.masked_fill(~mask, -1e9)                     # ignore padding positions
        weights = torch.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)  # (B, 2H)
        return context

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(emb_dim + 2 * hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + 2 * hidden_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def step(self, token, hidden, enc_out, mask):
        emb = self.dropout(self.embedding(token))                    # (B, E)
        context = self.attention(hidden, enc_out, mask)              # (B, 2H)
        out, hidden = self.gru(torch.cat([emb, context], dim=1).unsqueeze(1), hidden.unsqueeze(0))
        out, hidden = out.squeeze(1), hidden.squeeze(0)
        logits = self.fc_out(torch.cat([out, context, emb], dim=1))  # (B, V)
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim, dropout, pad_id)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim, dropout, pad_id)

    def forward(self, src, src_lens, tgt_in):
        # Teacher forcing: the decoder always receives the true previous target token
        mask = src != self.pad_id
        enc_out, hidden = self.encoder(src, src_lens)
        step_logits = []
        for t in range(tgt_in.size(1)):
            logits, hidden = self.decoder.step(tgt_in[:, t], hidden, enc_out, mask)
            step_logits.append(logits)
        return torch.stack(step_logits, dim=1)                       # (B, T, V)

    @torch.no_grad()
    def generate(self, src, src_lens, bos_id, eos_id, max_len):
        # Greedy decoding, batched; returns a list of token-id lists (without eos)
        mask = src != self.pad_id
        enc_out, hidden = self.encoder(src, src_lens)
        token = torch.full((src.size(0),), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        generated = []
        for _ in range(max_len):                                     # hard limit prevents infinite loops
            logits, hidden = self.decoder.step(token, hidden, enc_out, mask)
            logits[:, self.pad_id] = -1e9
            logits[:, bos_id] = -1e9
            token = logits.argmax(dim=1)
            generated.append(token)
            finished |= token == eos_id
            if finished.all():
                break
        generated = torch.stack(generated, dim=1).tolist()
        results = []
        for ids in generated:
            results.append(ids[:ids.index(eos_id)] if eos_id in ids else ids)
        return results
