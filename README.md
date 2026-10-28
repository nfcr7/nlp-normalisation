# Informal Text Normalization with BPE and Seq2Seq

**Project:** Normalisation de Texte Informel (Langage / Chat)
**Author:** Youssef Nafae
**Context:** University project, Master in Data Science and AI

---

## 1. What is this project?

People rarely write the way textbooks say they should. On Twitter and in chat apps, messages look like this:

| Informal input | Normalized output |
|---|---|
| `thaaaaaaaaank u` | `thank you` |
| `thx for the follow` | `thanks for the follow` |
| `rt @user : makin me happppy` | `rt @user : making me happy` |

This project builds an NLP system that **automatically converts informal English (tweets, chat) into standard English while preserving the meaning**.

The system is a **neural machine-translation-style model**: it is trained on a parallel corpus of (informal message, normalized message) pairs and learns the transformations by itself. **No hand-written dictionary of abbreviations is used.**

## 2. Why does it matter?

- **Most NLP tools expect clean text.** Machine translation, sentiment analysis, named-entity recognition, search and summarization are usually trained on standard language. Their accuracy drops sharply on slang, abbreviations and creative spelling. Normalization is a pre-processing step that makes all of them work better on social-media data.
- **Social media is a huge data source.** Opinion mining, public-health monitoring, crisis response and brand analysis all rely on noisy user-generated text.
- **Hand-written rules do not scale.** Slang evolves constantly and is context-dependent (`u` is `you`, but other abbreviations have several valid expansions). A learned model can adapt to new data instead of requiring endless manual rules.
- **It tackles a core NLP difficulty: the open vocabulary.** Elongated words (`happppy`), typos and invented spellings are practically infinite. Subword tokenization (BPE) lets the model handle words it has never seen.
- **Accessibility and readability.** Normalized text is easier for non-native speakers, screen readers and downstream tools to process.

## 3. How it works

```
informal message
      │
      ▼
BPE subword tokenizer  (learned on the training split only)
      │
      ▼
Bidirectional GRU encoder
      │
      ▼
Bahdanau attention  ──►  GRU decoder (autoregressive, greedy decoding)
      │
      ▼
BPE decoding + punctuation spacing  ──►  normalized message
```

1. **Data parsing.** The notebook inspects `/kaggle/input/`, detects the file format (token-per-line, alternating lines, or message-level TSV), and reconstructs `source_text` and `target_text` for every message. Malformed records are counted and skipped, never silently merged.
2. **Minimal preprocessing.** Only whitespace is normalized. Case, punctuation, mentions, hashtags, numbers and emoticons are kept, because the model must learn the dataset's conventions.
3. **BPE tokenization.** Byte-level BPE (Hugging Face Tokenizers), trained on the **train split only**. BPE is just a tokenization method: it splits text into subwords but does **not** itself normalize anything.
4. **Seq2Seq model (PyTorch).** Embeddings, bidirectional GRU encoder, additive attention with padding mask, GRU decoder. Trained with teacher forcing, padding-ignoring cross-entropy, gradient clipping, early stopping, and best-checkpoint saving based on **dev** loss.
5. **Inference.** `normalize_text(text)` tokenizes, decodes greedily until `<eos>` or a hard length limit, converts back to text and tidies punctuation.
6. **Evaluation.** Corpus-level SacreBLEU on the untouched **test** split, compared against a "copy the input" baseline, plus a leakage-aware score on test messages not seen in training.
7. **Error analysis.** Real test predictions grouped by phenomenon (elongation, abbreviations, ambiguity, mentions, punctuation), with automatically computed word-level explanations.

## 4. Repository contents

| File | Purpose |
|---|---|
| `informal_text_normalization_bpe_seq2seq.ipynb` | The complete Kaggle notebook (15 sections, from data discovery to final summary) |
| `app.py` | Streamlit demo (CPU-friendly, loads saved artifacts, never retrains) |
| `requirements.txt` | Dependencies for the demo |
| `README.md` | Hugging Face Spaces README (with the required YAML header) |
| `PROJECT_README.md` | This document |

Running the notebook produces, in `/kaggle/working/`:

```
normalizer_model/            model.pt, config.json, tokenizer.json, tokenizer_config.json,
                             seq2seq_model.py, inference.py, metrics.json, training_history.json
normalizer_model.zip         the artifacts, ready to download
hf_space/ and hf_space.zip   app.py + requirements.txt + README.md + normalizer_model/
outputs/                     private by-products (test predictions, French summary), not exported
```

## 5. How to run it

### On Kaggle (training)
1. Attach your private dataset to a new notebook (*Add Data*) and enable a GPU (*Settings → Accelerator*).
2. Upload and open `informal_text_normalization_bpe_seq2seq.ipynb`.
3. Run all cells in order. If file detection is ambiguous, the notebook stops with a clear message; set `FILE_OVERRIDES` in cell 3.1.
4. Use *Save Version → Save & Run All*, then download `hf_space.zip` from the **Output** tab.

### Locally (demo)
```bash
pip install -r requirements.txt
streamlit run app.py          # needs normalizer_model/ next to app.py
```

### Using the model from Python
```python
import sys
sys.path.insert(0, "normalizer_model")
from inference import normalize_text
print(normalize_text("idk what u mean lol"))
```

### Deploying on Hugging Face Spaces
Create a Streamlit Space, unzip `hf_space.zip` and upload its contents (including `normalizer_model/`). No API key is needed. For large `model.pt` files use the web upload, Git LFS, or a public model repo referenced through the `NORMALIZER_HF_REPO` Space variable. Full instructions are in section 14 of the notebook.

## 6. Design principles

- **Learn, don't hard-code.** All normalization behavior comes from the parallel corpus.
- **No leakage.** Tokenizer and model are trained on train only; dev is used only for model selection; test is used once at the end. Overlaps between splits are measured and reported.
- **Honest evaluation.** No score is written by hand. BLEU, error tables and the final summary are computed at run time. BLEU is paired with a copy baseline because it can reward mere copying and penalize valid rewordings.
- **One code path.** The model and inference code live in the exported modules, so the notebook, the ZIP and the Streamlit app all run identical code.
- **Readable and defendable.** Small model, simple PyTorch, commented steps, assertions with explicit error messages.

## 7. Testing status

The notebook was executed end to end in a sandbox on a **small synthetic dataset** (generated only to exercise the code), which verified parsing, tensor shapes, training, decoding, BLEU, error analysis, export and reload. It has **not** been run on the real private dataset, and **no real BLEU or accuracy figures exist yet**. All results appear only once you run the notebook on your data.

## 8. Limitations and future work

- Results depend on the size and annotation conventions of the dataset, including its inconsistencies.
- Context-dependent and ambiguous abbreviations remain hard.
- Greedy decoding propagates early mistakes; there is no explicit copy mechanism for rare mentions or URLs.
- BLEU does not fully capture semantic correctness.

**Ideas for improvement:** beam search, pointer-generator (copy) mechanism, data augmentation, larger or pretrained models, and human evaluation of meaning preservation.

## 9. Technologies

Python, PyTorch, Hugging Face Tokenizers, SacreBLEU, pandas, matplotlib, Streamlit, Hugging Face Spaces.

---
title: Informal Text Normalizer
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# Informal Text Normalizer

Normalisation de texte informel (Twitter / chat) : BPE (Hugging Face `tokenizers`) + Seq2Seq GRU avec attention (PyTorch).
Projet universitaire - Youssef Nafae.

## Structure attendue du Space

```
.
├── app.py
├── requirements.txt
├── README.md
└── normalizer_model/
    ├── best_model.pt
    ├── tokenizer.json
    ├── config.json
    ├── model_def.py
    └── inference.py
```

## Déploiement

1. Sur Kaggle, téléchargez `normalizer_model.zip` (ou `hf_space_package.zip`, qui contient déjà tout) depuis `/kaggle/working/` (onglet *Output*).
2. Créez un Space sur https://huggingface.co/new-space (SDK **Streamlit**, matériel CPU basique).
3. Dézippez, puis téléversez `app.py`, `requirements.txt`, `README.md` et le dossier `normalizer_model/` (onglet *Files* → *Add file* → *Upload files*).
4. Le Space se construit automatiquement. Aucune clé API n'est nécessaire et le jeu de données privé n'est pas utilisé.

## Fichiers volumineux

`best_model.pt` dépasse souvent 10 Mo : l'interface web de Hugging Face le gère automatiquement avec Git LFS. En ligne de commande :

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes app.py requirements.txt README.md normalizer_model
git commit -m "Add app and model"
git push
```

Alternative : publier les artefacts dans un **dépôt de modèle** Hugging Face et les télécharger au démarrage avec `huggingface_hub.snapshot_download`. Un dépôt public ne nécessite aucun jeton ; un dépôt privé exige un jeton de lecture stocké comme *secret* du Space.

## Test local

```bash
pip install -r requirements.txt
streamlit run app.py
```
