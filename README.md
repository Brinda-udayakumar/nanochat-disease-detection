# 🩺 NanoChat Disease Detection System

A conversational AI system for medical symptom analysis that combines a gradient-boosted disease classifier (XGBoost) with a transformer-based chatbot (nanochat) to enable natural-language symptom intake and disease prediction across **721 conditions**.

> ⚠️ **Disclaimer:** This is a research prototype and is **not** intended for medical diagnosis. For real medical concerns, consult a qualified physician. In emergencies, call your local emergency number (911 in the US, 108 in India).

---

## Results

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | **83.78%** |
| Top-3 Accuracy | **94.88%** |
| Top-5 Accuracy | **97.30%** |
| Macro F1 | 0.7405 |
| Weighted F1 | 0.8369 |
| Disease Classes | 721 |
| Symptom Features | 377 |
| Training Time | 190 seconds (GPU) |

---

## Setup

\`\`\`bash
git clone https://github.com/YOUR_USERNAME/nanochat-disease-detection.git
cd nanochat-disease-detection
pip install -r requirements.txt
\`\`\`

Download the dataset (see \`data/README.md\`), then:

\`\`\`bash
python src/train.py          # regenerates model/ artifacts
python src/chatbot_demo.py   # interactive chatbot (needs GPU)
\`\`\`

## Tech Stack

XGBoost, nanochat-d20 (561M param transformer via HuggingFace Transformers), PyTorch, custom NLP symptom extraction pipeline.

See \`notebooks/\` for the full training and evaluation workflow.
