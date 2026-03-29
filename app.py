from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from nlp_analyzer import SentimentAnalyzer
import requests as req
import re
import os

app = Flask(__name__)
CORS(app, origins=["*"])
analyzer = SentimentAnalyzer()

# ---------------------------------------------------------------------------
# URL Helpers
# ---------------------------------------------------------------------------

def resolve_short_url(url: str) -> str:
    """Expand short Amazon URLs like https://amzn.in/d/xxxxx to full URLs."""
    try:
        response = req.head(url, allow_redirects=True, timeout=10)
        return response.url
    except Exception:
        return url


def is_valid_amazon_url(url: str) -> bool:
    """Validate both full and short Amazon URLs."""
    full_pattern = r"https://(www\.)?amazon\.(in|com|co\.uk|de|fr|ca|com\.au|co\.jp)/.*"
    short_pattern = r"https://(www\.)?amzn\.(in|com|to)/.*"
    return bool(re.match(full_pattern, url) or re.match(short_pattern, url))


def get_reviews_page_url(url: str) -> str:
    """Try to get the dedicated reviews page for more reviews."""
    try:
        # Extract ASIN from URL
        asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
        if asin_match:
            asin = asin_match.group(1)
            # Extract domain
            domain_match = re.search(r'amazon\.(\w+(?:\.\w+)?)', url)
            domain = domain_match.group(0) if domain_match else 'amazon.in'
            return f"https://www.{domain}/product-reviews/{asin}?reviewerType=all_reviews&pageNumber=1"
    except Exception:
        pass
    return url


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def scrape_amazon_reviews(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    all_reviews = []
    product_name = "Unknown Product"
    product_image = None
    product_rating = "No Rating"

    try:
        # First load product page to get name, image, rating
        response = req.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, "lxml")

        product_name_tag = soup.find("span", {"id": "productTitle"})
        product_name = product_name_tag.get_text(strip=True) if product_name_tag else "Unknown Product"

        product_image_tag = soup.find("img", {"id": "landingImage"})
        product_image = product_image_tag.get("src") if product_image_tag else None

        product_rating_tag = soup.find("span", {"class": "a-icon-alt"})
        product_rating = product_rating_tag.get_text(strip=True) if product_rating_tag else "No Rating"

        # Get reviews from product page
        page_reviews = [r.get_text(strip=True) for r in soup.select("span[data-hook='review-body']")]
        all_reviews.extend(page_reviews)

        # Also scrape dedicated reviews page for more reviews
        reviews_url = get_reviews_page_url(url)
        if reviews_url != url:
            for page_num in range(1, 4):  # Scrape up to 3 pages of reviews
                paged_url = reviews_url.replace("pageNumber=1", f"pageNumber={page_num}")
                r2 = req.get(paged_url, headers=headers, timeout=15)
                soup2 = BeautifulSoup(r2.text, "lxml")
                page_reviews = [r.get_text(strip=True) for r in soup2.select("span[data-hook='review-body']")]
                if not page_reviews:
                    break
                all_reviews.extend(page_reviews)

        # Deduplicate reviews
        all_reviews = list(dict.fromkeys(all_reviews))

    except Exception as e:
        print(f"Scraping error: {e}")

    return all_reviews, product_name, product_image, product_rating


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field in request body."}), 400

    url = data["url"].strip()

    # Resolve short URLs like amzn.in/d/xxxxx
    if "amzn.in" in url or "amzn.com" in url or "amzn.to" in url:
        url = resolve_short_url(url)
        print(f"Resolved short URL to: {url}")

    # Clean URL — remove query parameters that may cause issues
    url_clean = re.sub(r'\?.*$', '', url)
    if is_valid_amazon_url(url_clean):
        url = url_clean

    if not is_valid_amazon_url(url):
        return jsonify({"error": "Invalid URL. Please provide a valid Amazon product URL (amazon.in, amazon.com, or amzn.in short links)."}), 422

    reviews, product_name, product_image, product_rating = scrape_amazon_reviews(url)

    if not reviews:
        return jsonify({"error": "No reviews found. Amazon may have blocked the request or this product has no reviews yet."}), 404

    analyzed_reviews = analyzer.analyze_reviews(reviews)

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for review in analyzed_reviews:
        counts[review["sentiment"]] += 1

    return jsonify({
        "product_name": product_name,
        "product_image": product_image,
        "product_rating": product_rating,
        "total_reviews": len(analyzed_reviews),
        "counts": counts,
        "reviews": analyzed_reviews,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
