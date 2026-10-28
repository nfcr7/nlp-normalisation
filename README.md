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

1. Téléchargez `normalizer_model.zip`  depuis giuthb .
2. Dézippez, puis téléversez `app.py`, `requirements.txt`, `README.md` .
4. puis lance streamlite les terminal qui contient ce dossier .

## Fichiers volumineux

`best_model.pt` dépasse souvent 10 Mo : l'interface web de Hugging Face le gère automatiquement avec Git LFS. En ligne de commande :

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes app.py requirements.txt README.md normalizer_model
git commit -m "Add app and model"
git push
```

## Test local

```bash
pip install -r requirements.txt
streamlit run app.py
```
