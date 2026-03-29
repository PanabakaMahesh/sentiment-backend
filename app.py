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
# ScraperAPI config — free tier: 1000 credits/month
# Sign up at scraperapi.com and paste your key here OR set as env variable
# ---------------------------------------------------------------------------
SCRAPER_API_KEY = os.environ.get("SCRAPER_API_KEY", "")

AMAZON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def build_scraper_url(target_url: str) -> str:
    """Wrap target URL with ScraperAPI if key is available."""
    if SCRAPER_API_KEY:
        return f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={req.utils.quote(target_url)}&render=false&country_code=in"
    return target_url


def fetch_page(url: str, session: req.Session) -> BeautifulSoup:
    """Fetch a page via ScraperAPI or directly."""
    fetch_url = build_scraper_url(url)
    response = session.get(fetch_url, headers=AMAZON_HEADERS, timeout=60)
    print(f"  Status: {response.status_code} | Length: {len(response.text)}")
    return BeautifulSoup(response.text, "lxml")


# ---------------------------------------------------------------------------
# URL Helpers
# ---------------------------------------------------------------------------

def resolve_short_url(url: str) -> str:
    try:
        session = req.Session()
        response = session.get(url, headers=AMAZON_HEADERS, allow_redirects=True, timeout=15)
        return response.url
    except Exception as e:
        print(f"URL resolve error: {e}")
        return url


def is_valid_amazon_url(url: str) -> bool:
    full_pattern = r"https://(www\.)?amazon\.(in|com|co\.uk|de|fr|ca|com\.au|co\.jp)/.*"
    short_pattern = r"https?://(www\.)?amzn\.(in|com|to)/.*"
    return bool(re.match(full_pattern, url) or re.match(short_pattern, url))


def get_asin_from_url(url: str) -> str:
    patterns = [
        r'/dp/([A-Z0-9]{10})',
        r'/product/([A-Z0-9]{10})',
        r'/gp/product/([A-Z0-9]{10})',
        r'ASIN=([A-Z0-9]{10})',
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

def scrape_amazon_reviews(url: str):
    all_reviews = []
    product_name = "Unknown Product"
    product_image = None
    product_rating = "No Rating"

    session = req.Session()

    try:
        print(f"Fetching product page: {url}")
        soup = fetch_page(url, session)

        # Extract product details
        product_name_tag = soup.find("span", {"id": "productTitle"})
        product_name = product_name_tag.get_text(strip=True) if product_name_tag else "Unknown Product"

        product_image_tag = (
            soup.find("img", {"id": "landingImage"}) or
            soup.find("img", {"id": "imgBlkFront"})
        )
        product_image = product_image_tag.get("src") if product_image_tag else None

        product_rating_tag = soup.find("span", {"class": "a-icon-alt"})
        product_rating = product_rating_tag.get_text(strip=True) if product_rating_tag else "No Rating"

        # Reviews from product page
        product_page_reviews = [r.get_text(strip=True) for r in soup.select("span[data-hook='review-body']")]
        all_reviews.extend(product_page_reviews)
        print(f"Product page reviews: {len(product_page_reviews)}")

        # Dedicated reviews pages
        asin = get_asin_from_url(url)
        domain = get_domain_from_url(url)

        if asin:
            print(f"ASIN: {asin} | Domain: {domain}")
            for page_num in range(1, 6):
                reviews_url = (
                    f"https://www.{domain}/product-reviews/{asin}"
                    f"?reviewerType=all_reviews&pageNumber={page_num}&sortBy=recent"
                )
                print(f"Fetching reviews page {page_num}...")
                time.sleep(1)
                try:
                    soup2 = fetch_page(reviews_url, session)
                    page_reviews = [
                        r.get_text(strip=True)
                        for r in soup2.select("span[data-hook='review-body']")
                    ]
                    print(f"  Page {page_num}: {len(page_reviews)} reviews")
                    if not page_reviews:
                        break
                    all_reviews.extend(page_reviews)
                except Exception as e:
                    print(f"  Page {page_num} error: {e}")
                    break

        # Deduplicate
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
    print(f"\n=== New request: {url} ===")

    # Resolve short URLs
    if re.search(r'amzn\.(in|com|to)', url):
        url = resolve_short_url(url)
        print(f"Resolved to: {url}")

    # Strip tracking params
    clean_url = re.sub(r'\?.*$', '', url)
    if is_valid_amazon_url(clean_url):
        url = clean_url

    if not is_valid_amazon_url(url):
        return jsonify({
            "error": f"Invalid Amazon URL. Please use amazon.in, amazon.com, or amzn.in short links."
        }), 422

    reviews, product_name, product_image, product_rating = scrape_amazon_reviews(url)

    if not reviews:
        return jsonify({
            "error": "No reviews found. Amazon blocked the scraping request. Please try again in a few minutes or try a different product."
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
