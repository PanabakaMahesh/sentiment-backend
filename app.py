from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from nlp_analyzer import SentimentAnalyzer
import requests as req
import re
import os
import time

app = Flask(__name__)
CORS(app, origins=["*"])
analyzer = SentimentAnalyzer()

# ---------------------------------------------------------------------------
# Fixed realistic browser headers — consistent and reliable
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# ---------------------------------------------------------------------------
# URL Helpers
# ---------------------------------------------------------------------------

def resolve_short_url(url: str) -> str:
    """Expand short Amazon URLs like https://amzn.in/d/xxxxx to full URLs."""
    try:
        response = req.get(url, headers=HEADERS, allow_redirects=True, timeout=15)
        return response.url
    except Exception as e:
        print(f"URL resolve error: {e}")
        return url


def is_valid_amazon_url(url: str) -> bool:
    full_pattern = r"https://(www\.)?amazon\.(in|com|co\.uk|de|fr|ca|com\.au|co\.jp)/.*"
    short_pattern = r"https?://(www\.)?amzn\.(in|com|to)/.*"
    return bool(re.match(full_pattern, url) or re.match(short_pattern, url))


def get_asin_from_url(url: str) -> str:
    """Extract ASIN from Amazon URL."""
    patterns = [
        r'/dp/([A-Z0-9]{10})',
        r'/product/([A-Z0-9]{10})',
        r'/gp/product/([A-Z0-9]{10})',
        r'/product-reviews/([A-Z0-9]{10})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_domain_from_url(url: str) -> str:
    match = re.search(r'(amazon\.\w+(?:\.\w+)?)', url)
    return match.group(1) if match else 'amazon.in'


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def scrape_page(url: str, session: req.Session) -> BeautifulSoup:
    time.sleep(1.5)  # Fixed polite delay — consistent and avoids rate limiting
    response = session.get(url, headers=HEADERS, timeout=20)
    print(f"Scraped {url} — status {response.status_code}")
    return BeautifulSoup(response.text, "lxml")


def scrape_amazon_reviews(url: str):
    all_reviews = []
    product_name = "Unknown Product"
    product_image = None
    product_rating = "No Rating"

    session = req.Session()
    session.headers.update(HEADERS)

    try:
        # Step 1: Load product page for metadata
        soup = scrape_page(url, session)

        product_name_tag = soup.find("span", {"id": "productTitle"})
        product_name = product_name_tag.get_text(strip=True) if product_name_tag else "Unknown Product"

        product_image_tag = (
            soup.find("img", {"id": "landingImage"}) or
            soup.find("img", {"id": "imgBlkFront"}) or
            soup.find("img", {"data-old-hires": True})
        )
        product_image = product_image_tag.get("src") if product_image_tag else None

        product_rating_tag = soup.find("span", {"class": "a-icon-alt"})
        product_rating = product_rating_tag.get_text(strip=True) if product_rating_tag else "No Rating"

        # Collect reviews from product page
        page_reviews = [r.get_text(strip=True) for r in soup.select("span[data-hook='review-body']")]
        all_reviews.extend(page_reviews)
        print(f"Product page: {len(page_reviews)} reviews")

        # Step 2: Scrape dedicated reviews pages for more data
        asin = get_asin_from_url(url)
        domain = get_domain_from_url(url)

        if asin:
            for page_num in range(1, 6):  # Up to 5 pages = ~50 reviews
                reviews_url = (
                    f"https://www.{domain}/product-reviews/{asin}"
                    f"?reviewerType=all_reviews&pageNumber={page_num}&sortBy=recent"
                )
                try:
                    soup2 = scrape_page(reviews_url, session)

                    # Check if Amazon served a captcha/block page
                    if soup2.find("form", {"action": "/errors/validateCaptcha"}):
                        print(f"Amazon served captcha on page {page_num}, stopping.")
                        break

                    page_reviews = [
                        r.get_text(strip=True)
                        for r in soup2.select("span[data-hook='review-body']")
                    ]
                    print(f"Reviews page {page_num}: {len(page_reviews)} reviews")

                    if not page_reviews:
                        print(f"No reviews on page {page_num}, stopping.")
                        break

                    all_reviews.extend(page_reviews)

                except Exception as e:
                    print(f"Error on reviews page {page_num}: {e}")
                    break

        # Deduplicate while preserving order
        seen = set()
        unique_reviews = []
        for r in all_reviews:
            r_clean = r.strip()
            if r_clean and r_clean not in seen:
                seen.add(r_clean)
                unique_reviews.append(r_clean)

        print(f"Total unique reviews: {len(unique_reviews)}")
        return unique_reviews, product_name, product_image, product_rating

    except Exception as e:
        print(f"Scraping error: {e}")
        return [], product_name, product_image, product_rating


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field in request body."}), 400

    url = data["url"].strip()
    print(f"Received URL: {url}")

    # Resolve short URLs like amzn.in/d/xxxxx
    if re.search(r'amzn\.(in|com|to)', url):
        print(f"Resolving short URL...")
        url = resolve_short_url(url)
        print(f"Resolved to: {url}")

    # Strip tracking query parameters for cleaner scraping
    clean_url = re.sub(r'\?.*$', '', url)
    if is_valid_amazon_url(clean_url):
        url = clean_url
        print(f"Cleaned URL: {url}")

    if not is_valid_amazon_url(url):
        return jsonify({
            "error": f"Invalid URL. Please provide a valid Amazon product URL (amazon.in, amazon.com or amzn.in short links)."
        }), 422

    reviews, product_name, product_image, product_rating = scrape_amazon_reviews(url)

    if not reviews:
        return jsonify({
            "error": "No reviews found. Amazon may have blocked the request. Please try again in a few minutes or use a different product URL."
        }), 404

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
