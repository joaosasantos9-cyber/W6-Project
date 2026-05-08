import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "roboreviews_app_cache.json"
META_RULES = [
    ("Kindle / E-Reader", ["kindle", "e-reader", "ebook", "paperwhite"]),
    ("Tablet", ["fire tablet", "tablet"]),
    ("Smart Home & Audio", ["echo", "alexa", "fire tv", "speaker", "audio", "headphone", "bluetooth"]),
    ("Accessories / Batteries", [
        "battery", "batteries", "charger", "power", "usb", "adapter",
        "case", "keyboard", "stand", "mount", "cable", "screen protector",
    ]),
]

def load_cache() -> dict:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            "roboreviews_app_cache.json is missing. Recreate it from the notebook outputs before running the app."
        )
    with open(CACHE_PATH, "r") as f:
        return json.load(f)


CACHE = load_cache()
CATEGORIES = CACHE.get("categories", [])
ARTICLES = CACHE.get("final_articles", {})
METRICS = CACHE.get("metrics", {})

def df(rows):
    return pd.DataFrame(rows or [])


def pct(value):
    if value is None or value == "N/A":
        return "N/A"
    return f"{float(value):.1f}%"


def score(value):
    if value is None or value == "N/A":
        return "N/A"
    return f"{float(value):.3f}"



def quick_sentiment(text: str) -> tuple[str, str]:
    text_l = text.lower()
    strong_negative = [
        "stopped working", "does not work", "didn't work", "waste of money",
        "complete waste", "very disappointed", "broken", "failed", "returning",
    ]
    if any(term in text_l for term in strong_negative):
        return "Negative", "negative"

    positive_terms = [
        "love", "great", "excellent", "perfect", "amazing", "best", "good",
        "happy", "works well", "recommend", "clear", "comfortable", "crystal clear",
    ]
    negative_terms = [
        "bad", "poor", "terrible", "waste", "broken", "stopped", "disappointed",
        "awful", "return", "problem", "issue", "loose", "failed", "worse", "poor quality",
    ]
    pos = sum(term in text_l for term in positive_terms)
    neg = sum(term in text_l for term in negative_terms)
    if neg > pos:
        return "Negative", "negative"
    if pos > neg:
        return "Positive", "positive"
    return "Neutral", "neutral"

def predict_sentiment(review_text: str) -> tuple[str, str]:
    if not review_text.strip():
        return "N/A", "neutral"
    return quick_sentiment(review_text)

def assign_category(text: str) -> str:
    text_l = text.lower()
    for category, keywords in META_RULES:
        if any(keyword in text_l for keyword in keywords):
            return category
    return CATEGORIES[0] if CATEGORIES else "Other / General Electronics"


def top_products(category: str):
    return CACHE.get("top_products", {}).get(category, [])


def article_for_category(category: str) -> str:
    return ARTICLES.get(category, "No saved article found for this category.")


def products_for_category(category: str):
    return CACHE.get("product_rows", {}).get(category, [])


def reviews_for_category(category: str, sentiment: str, limit: int):
    rows = CACHE.get("sample_reviews", {}).get(category, {}).get(sentiment, [])
    return rows[: int(limit)]


def analyze_review(review_text: str):
    if not review_text.strip():
        return "<strong>Please enter a review.</strong>", "", df([]), ""

    sentiment_text, sentiment_label = predict_sentiment(review_text)
    category = assign_category(review_text)
    color = {
        "positive": "#166534",
        "neutral": "#92400e",
        "negative": "#991b1b",
    }.get(sentiment_label, "#1f2937")
    sentiment_html = (
        f'<div class="result-card"><span>Predicted sentiment</span>'
        f'<strong style="color:{color}">{sentiment_text}</strong></div>'
    )
    return sentiment_html, category, df(top_products(category)), article_for_category(category)


def category_view(category: str, sentiment: str, limit: int):
    return (
        df(top_products(category)),
        article_for_category(category),
        df(products_for_category(category)),
        df(reviews_for_category(category, sentiment, limit)),
    )


CSS = """
.gradio-container {max-width: 1180px !important;}
.result-card {border:1px solid #334155; border-radius:8px; padding:14px; background:#111827;}
.result-card span {display:block; font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:.04em;}
.result-card strong {display:block; font-size:24px; margin-top:4px;}
"""

EXAMPLES = [
    ["I absolutely love this Kindle. The battery lasts for weeks and the screen is crystal clear."],
    ["The Echo Dot stopped working after 2 weeks. Very disappointed with the build quality."],
    ["These AA batteries are okay for the price. They last a reasonable amount of time but nothing special."],
    ["The USB-C cable is decent but the connector feels loose. Works fine but I expected better quality."],
]


with gr.Blocks(title="RoboReviews") as demo:
    gr.Markdown("# RoboReviews Product Review App")
    gr.Markdown("Fast demo mode uses precomputed project outputs for a stable deployment demo.")

    with gr.Tab("Live Review"):
        with gr.Row():
            with gr.Column(scale=1):
                review_input = gr.Textbox(label="Product review", placeholder="Paste or type a product review here...", lines=6)
                analyze_btn = gr.Button("Analyze Review", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=review_input, label="Examples")
            with gr.Column(scale=1):
                sentiment_out = gr.HTML(label="Sentiment")
                category_out = gr.Textbox(label="Detected category", interactive=False)
                live_top_products = gr.Dataframe(label="Top 3 Products in Detected Category", interactive=False)
        live_article = gr.Markdown(label="Recommendation article")
        analyze_btn.click(
            analyze_review,
            inputs=[review_input],
            outputs=[sentiment_out, category_out, live_top_products, live_article],
        )

    with gr.Tab("Category Explorer"):
        with gr.Row():
            category_input = gr.Dropdown(CATEGORIES, value=CATEGORIES[0] if CATEGORIES else None, label="Category")
            sentiment_filter = gr.Dropdown(["All", "positive", "neutral", "negative"], value="All", label="Review sentiment")
            review_limit = gr.Slider(5, 50, value=10, step=5, label="Review rows")
        top_products_out = gr.Dataframe(label="Top 3 Products", interactive=False)
        article_output = gr.Markdown(label="Saved Recommendation Article")
        with gr.Row():
            product_output = gr.Dataframe(label="Products in Category", interactive=False)
            review_output = gr.Dataframe(label="Sample Reviews", interactive=False)
        refresh_btn = gr.Button("Refresh Category", variant="primary")
        refresh_btn.click(
            category_view,
            inputs=[category_input, sentiment_filter, review_limit],
            outputs=[top_products_out, article_output, product_output, review_output],
        )

    with gr.Tab("Project Results"):
        gr.Dataframe(
            value=df(CACHE.get("model_scores", [])),
            label="Task 1 Model Scores",
            interactive=False,
        )
        gr.Dataframe(
            value=df(CACHE.get("category_stats", [])),
            label="Category Summary with Top 3 Products",
            interactive=False,
        )

    with gr.Tab("Evaluation"):
        gr.Dataframe(
            value=df(CACHE.get("rouge_rows", [])),
            label="Task 3 ROUGE-L by Category",
            interactive=False,
        )
        gr.Dataframe(
            value=df(CACHE.get("quality_rows", [])),
            label="Automatic Article Requirement Checks",
            interactive=False,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("SHARE", "false").lower() == "true",
        theme=gr.themes.Soft(),
        css=CSS,
        show_error=True,
    )
