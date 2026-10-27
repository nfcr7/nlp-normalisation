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
