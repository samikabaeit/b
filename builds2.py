#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import csv
import re

def sanitize_filename(name):
    """Sanitize a string to be used as a safe filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

# Define the range of build IDs to scrape
build_start = 502834
build_end = 502940

# Define the expected headers for the table rows
headers = ["קטגוריה", "מק\"ט", "מוצרים", "מחיר", "כמות", "סה\"כ"]

for build_id in range(build_start, build_end + 1):
    url = f"https://tms.co.il/builds/{build_id}"
    print(f"Scraping build: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        continue

    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract the build's price from the <span class="price-format"> element.
    price_elem = soup.find("span", class_="price-format")
    if not price_elem:
        print(f"Price element not found on {url}")
        continue

    # Remove the currency symbol and extra whitespace to get the numeric price.
    raw_price = price_elem.get_text(strip=True)
    price = raw_price.replace("₪", "").strip()
    
    # Use build ID in filename in case more than one build has the same price.
    filename = sanitize_filename(f"{price}") + ".csv"

    # Find the table container (div with class "table-build-preview__body")
    table_div = soup.find("div", class_="table-build-preview__body")
    if not table_div:
        print(f"No table found on {url}")
        continue

    # Find all rows within the table.
    rows = table_div.find_all("div", class_="table-build-preview__row")
    if not rows:
        print(f"No rows found in the table on {url}")
        continue

    # Prepare the list to hold CSV rows.
    csv_rows = []
    
    # Loop over each row and extract the columns.
    for row in rows:
        # We assume each cell is in a div with class "table-build-preview__column".
        cells = row.find_all("div", class_="table-build-preview__column")
        if len(cells) < 6:
            # Skip rows that do not have enough columns.
            continue

        # Extract text for the 6 columns.
        # For the price cell, we get the text outside any inner span.
        category = cells[0].get_text(strip=True)
        product_model = cells[1].get_text(strip=True)
        product_name = cells[2].get_text(strip=True)

        # For the 'מחיר' column, take the direct text of the cell.
        price_cell = cells[3]
        price_text = price_cell.find(text=True, recursive=False)
        price_col = price_text.strip() if price_text else price_cell.get_text(strip=True)

        quantity = cells[4].get_text(strip=True)

        # For the 'סה"כ' column, try to get the text inside the inner span if available.
        total_cell = cells[5]
        total_span = total_cell.find("span")
        total = total_span.get_text(strip=True) if total_span else total_cell.get_text(strip=True)

        csv_rows.append([category, product_model, product_name, price_col, quantity, total])

    # Save the scraped rows into a CSV file with the name based on the build's price.
    if csv_rows:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(csv_rows)
        print(f"Wrote {len(csv_rows)} rows to {filename}")
    else:
        print(f"No valid rows found for build {build_id}")

print("Scraping and CSV generation completed.")
