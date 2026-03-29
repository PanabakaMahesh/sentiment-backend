from flask import Flask, request, jsonify
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from nlp_analyzer import SentimentAnalyzer
import time
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter mobile client
analyzer = SentimentAnalyzer()


def is_valid_amazon_url(url: str) -> bool:
    """Validate that the URL is a recognised Amazon product page."""
    pattern = r"https://(www\.)?amazon\.(in|com|co\.uk|de|fr|ca|com\.au|co\.jp)/.*"
    return bool(re.match(pattern, url))


def scrape_amazon_reviews(url: str):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)
        time.sleep(3)

        # Scroll to trigger lazy-loaded content
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "lxml")

        reviews = [
            r.get_text(strip=True)
            for r in soup.select("span[data-hook='review-body']")
        ]

        product_name_tag = soup.find("span", {"id": "productTitle"})
        product_name = (
            product_name_tag.get_text(strip=True) if product_name_tag else "Unknown Product"
        )

        product_image_tag = soup.find("img", {"id": "landingImage"})
        product_image = product_image_tag.get("src") if product_image_tag else None

        product_rating_tag = soup.find("span", {"class": "a-icon-alt"})
        product_rating = (
            product_rating_tag.get_text(strip=True) if product_rating_tag else "No Rating"
        )

        return reviews, product_name, product_image, product_rating

    except Exception as e:
        print(f"Scraping error: {e}")
        return [], "Unknown Product", None, "No Rating"
    finally:
        driver.quit()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accepts JSON body: { "url": "<amazon_product_url>" }

    Returns JSON:
    {
        "product_name": str,
        "product_image": str | null,
        "product_rating": str,
        "total_reviews": int,
        "counts": { "positive": int, "neutral": int, "negative": int },
        "reviews": [
            { "text": str, "sentiment": str, "confidence": float },
            ...
        ]
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field in request body."}), 400

    url = data["url"].strip()

    if not is_valid_amazon_url(url):
        return jsonify({"error": "Invalid URL. Please provide a valid Amazon product URL."}), 422

    reviews, product_name, product_image, product_rating = scrape_amazon_reviews(url)

    if not reviews:
        return jsonify({"error": "No reviews found. The product page may not have reviews, or Amazon blocked the request."}), 404

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
    """Simple liveness probe used by the Flutter app on startup."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
