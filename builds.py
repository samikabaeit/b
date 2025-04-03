#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import csv
import re
import os

def sanitize_filename(name):
    """Sanitize category name to be used as a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

# Range of build IDs to scrape
build_start = 502840
build_end = 502940

# Define header names (in the order of the table columns)
headers = ["קטגוריה", "מק\"ט", "מוצרים", "מחיר", "כמות", "סה\"כ"]

# Dictionary to accumulate rows for each category
category_data = {}

# Iterate over each build page
for build_id in range(build_start, build_end + 1):
    url = f"https://tms.co.il/builds/{build_id}"
    print(f"Scraping URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        continue

    soup = BeautifulSoup(response.content, "html.parser")
    table_div = soup.find("div", class_="table-build-preview__body")
    if not table_div:
        print(f"No table found at {url}")
        continue

    # Find all rows; each row is a div with class "table-build-preview__row"
    rows = table_div.find_all("div", class_="table-build-preview__row")
    if not rows:
        print(f"No rows found in the table at {url}")
        continue

    for row in rows:
        # Find all columns in this row
        cells = row.find_all("div", class_="table-build-preview__column")
        if len(cells) < 6:
            continue  # Skip rows that don't have all 6 expected columns

        # Extract data from each cell
        # 1. קטגוריה
        category = cells[0].get_text(strip=True)
        # 2. מק"ט (Product Model)
        product_model = cells[1].get_text(strip=True)
        # 3. מוצרים (Product Name)
        product_name = cells[2].get_text(strip=True)
        # 4. מחיר (Price) - get only the direct text (ignoring the span)
        price_cell = cells[3]
        price_text = price_cell.find(text=True, recursive=False)
        price = price_text.strip() if price_text else price_cell.get_text(strip=True)
        # 5. כמות (Quantity)
        quantity = cells[4].get_text(strip=True)
        # 6. סה"כ (Total) - if there is a span, get its text
        total_cell = cells[5]
        total_span = total_cell.find("span")
        total = total_span.get_text(strip=True) if total_span else total_cell.get_text(strip=True)

        row_data = [category, product_model, product_name, price, quantity, total]

        # Group rows by category name
        if category not in category_data:
            category_data[category] = []
        category_data[category].append(row_data)

# Create a CSV file for each category
for category, rows in category_data.items():
    filename = sanitize_filename(category) + ".csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {filename}")

print("Scraping and CSV generation completed.")
