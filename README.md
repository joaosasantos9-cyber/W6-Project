# W6-Project
RoboReviews
An end-to-end NLP pipeline that turns raw Amazon product reviews into ready-to-publish recommendation articles.
Three connected models — sentiment classification, product clustering, and generative article writing — running on top of a single small base LLM (Qwen 2.5 1.5B) fine-tuned twice with LoRA. The whole thing runs on a laptop.

Pipeline at a glance
Raw reviews  ─►  Preprocess  ─►  Task 1: Sentiment  ─►  Task 2: Clusters  ─►  Task 3: Articles  ─►  Final outputs
  46,538            clean +      Qwen + LoRA           Embeddings + KMeans   Qwen + LoRA          JSON + DOCX
                    split        (75% accuracy)        (K=4, silhouette 0.16) (self-distillation)
Each stage's output is a checked input to the next. Every handoff has an assert so a broken upstream step fails loudly.


Setup
Requirements

Python 3.10+
CPU works for everything except full-corpus inference (use a GPU there)
~6 GB free disk for model downloads (Qwen 2.5 1.5B + Sentence-Transformer)

Data
The notebook expects three CSV files in the working directory (or in data/ — adjust FILES in cell 9):

1429_1.csv
Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv

All three are from the public Kaggle dataset:
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products

What's inside each task
Task 1 — Sentiment classification

Input: cleaned review text from the test set
Output: one of positive, neutral, negative
Three modes evaluated on a stratified 300-row sample (100 per class):

Zero-shot — single prompt, no examples
N-shot — 8 in-context examples covering all three classes
LoRA fine-tune — r=8, target q/k/v/o_proj, balanced 1,500-sample training set (500 per class)


Evaluation: accuracy, weighted F1, per-class precision/recall/F1, McNemar's test, Wilson 95% CI
Output column reused downstream: task1_sentiment — LoRA prediction where available, rating-derived fallback otherwise

Task 2 — Category clustering

Input: 100 unique product names (deduplicated)
Output: cluster ID per product, then propagated back to all 46,538 reviews via product-name join (with meta-category fallback for unmatched rows)
Two methods compared at K = 3…7:

Option A: TF-IDF (5,000 features) + KMeans
Option B: Sentence Embeddings (all-MiniLM-L6-v2, 384-d) + KMeans


Selected: K=4 with sentence embeddings — picked over the math-best K=7 because the project brief asks for interpretable meta-categories
Final categories: Accessories / Batteries, Kindle / E-Reader, Tablet, Smart Home & Audio

Task 3 — Article generation

Input: cluster profile (top products, sample positive reviews, sample complaints)
Output: a recommendation article per category — Best products, Key differences, Common complaints
Self-distillation: the base Qwen generates a long-prompted reference article per cluster; the LoRA adapter (r=16) is then fine-tuned to produce that article from a shorter prompt
Evaluation: ROUGE-1/2/L against held-out test reviews — never reusing the prompt input as the reference, to avoid circularity


Anti-leakage and reproducibility

train_test_split is stratified on sentiment, before any modelling
Explicit duplicate check on clean_text across train/test (zero overlap, asserted)
Handoff asserts after each stage: train + test = full dataset; cluster IDs joined to reviews; etc.
All seeds fixed: SEED = 42 propagated to numpy, torch.manual_seed, every random_state, and the SFT trainer
Caveat: do_sample=True is used in article generation. To make Task 3 fully reproducible, switch to do_sample=False (greedy) or call transformers.set_seed(SEED) before each generation


Deliverables

Readme file
Notebook: notebooks/project1_finalV3_improved_final.ipynb — runs end-to-end
Articles: outputs/final_articles.json — the four recommendation articles
Presentation: deliverables/RoboReviews_project.pptx
