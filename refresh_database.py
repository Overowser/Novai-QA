import aiohttp
from bs4 import BeautifulSoup
import asyncio
import re
from datetime import datetime
import psycopg2
import os
from dotenv import load_dotenv
from fake_headers import Headers

header = Headers(browser="chrome", os="win", headers=True)

load_dotenv()
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_USER = os.getenv("PG_USER")
PG_DB = os.getenv("PG_DB")

async def refresh_database(keyword):
    print(f"Starting refresh_database with keyword: {keyword}")
    headers = header.generate()
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        novel_title, novel_image, chapter_titles, chapter_urls = await get_urls(session, keyword)
        print(f"Retrieved novel title: {novel_title}, with {len(chapter_titles)} chapters")

    conn = psycopg2.connect(
        host=PG_HOST,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
        port="5432"
    )
    cursor = conn.cursor()

    while True:
        try:
            urls_to_scrape = []
            urls_to_update = []
            for url, title in zip(chapter_urls, chapter_titles):
                cursor.execute("SELECT * FROM chapters WHERE chapter_url = %s", (url,))
                result = cursor.fetchone()

                if result:
                    if "security by Cloudflare" in result[5]:
                        urls_to_update.append((url, title))
                    else:
                        continue
                else:
                    urls_to_scrape.append((url, title))
            
            print(f"URLs to scrape: {len(urls_to_scrape)}, URLs to update: {len(urls_to_update)}")

            headers = header.generate()
            connector = aiohttp.TCPConnector(limit=20)
            async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
                await chapter_to_db(session, novel_title, novel_image, urls_to_scrape, urls_to_update, cursor, conn)
                break
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    cursor.close()
    conn.close()
    print("Database refresh completed")
    return

async def chapter_to_db(session, novel_title, novel_image, urls_to_scrape, urls_to_update, cursor, conn):
    print(f"Processing novel: {novel_title}")
    cursor.execute("SELECT * FROM novels WHERE novel_title = %s", (novel_title,))
    result = cursor.fetchone()

    if result:
        novel_id = result[0]
    else:
        cursor.execute(
            "INSERT INTO novels (novel_title, novel_image) VALUES (%s, %s) RETURNING id",
            (novel_title, novel_image)
        )
        novel_id = cursor.fetchone()[0]
        conn.commit()
    print(f"Novel ID: {novel_id}")

    chapter_urls = [url[0] for url in urls_to_scrape]
    chapter_titles = [url[1] for url in urls_to_scrape]

    tasks = [get_page_content(session, f"https://novelfull.com{chapter_url}") for chapter_url in chapter_urls]
    chapter_contents = await asyncio.gather(*tasks)
    chapter_contents = [preprocess(text) for text in chapter_contents]

    for chapter_title, chapter_url, chapter_content in zip(chapter_titles, chapter_urls, chapter_contents):
        chapter_number = re.search(r'\d+', chapter_title)
        chapter_number = int(chapter_number.group())

        cursor.execute(
            "INSERT INTO chapters (novel_id, chapter_number, chapter_title, chapter_url, chapter_content) VALUES (%s, %s, %s, %s, %s)",
            (novel_id, chapter_number, chapter_title, chapter_url, chapter_content)
        )
        conn.commit()
        print(f"Inserted chapter {chapter_title} into database with number {chapter_number}")

    chapter_urls = [url[0] for url in urls_to_update]
    chapter_titles = [url[1] for url in urls_to_update]

    tasks = [get_page_content(session, f"https://novelfull.com{chapter_url}") for chapter_url in chapter_urls]
    chapter_contents = await asyncio.gather(*tasks)
    chapter_contents = [preprocess(text) for text in chapter_contents]

    for chapter_title, chapter_content in zip(chapter_titles, chapter_contents):
        chapter_number = re.search(r'\d+', chapter_title)
        chapter_number = int(chapter_number.group())
        cursor.execute(
            "UPDATE chapters SET chapter_content = %s WHERE chapter_number = %s;",
            (chapter_content, chapter_number)
        )
        conn.commit()
        print(f"Updated chapter {chapter_title} in database with number {chapter_number}")

    return

async def fetch_html(session, url):
    print(f"Fetching HTML for URL: {url}")
    async with session.get(url) as response:
        html = await response.text()
        return BeautifulSoup(html, 'html.parser')

async def get_page_content(session, url):
    soup = await fetch_html(session, url)
    text = '\n'.join(
        [
            item.text
            for item in soup.select('p')
            if item.text.strip() and 'translator' not in item.text.lower() and 'copyright' not in item.text.lower()
        ]
    )
    return text

async def get_urls(session, keyword):
    print(f"Searching for keyword: {keyword}")
    search_url = f"https://novelfull.com/search?keyword={keyword}"
    search_soup = await fetch_html(session, search_url)

    novel_title = search_soup.select_one('.truyen-title a').text

    chapter_list_url = f"https://novelfull.com{search_soup.select_one('.truyen-title a')['href']}"
    chapter_list_soup = await fetch_html(session, chapter_list_url)

    last_url = f'https://novelfull.com{chapter_list_soup.select_one("li.last a")["href"]}'
    last_page = int(last_url.split("=")[-1])
    url_part1 = last_url.split("page=")[0] + "page="

    tasks = [
        fetch_html(session, f"{url_part1}{page_num}")
        for page_num in range(1, last_page+1)
    ]
    page_soups = await asyncio.gather(*tasks)

    chapter_titles = []
    chapter_urls = []

    for soup in page_soups:
        chapter_titles += [item.text for item in soup.select('#list-chapter .row li a')]
        chapter_urls += [item['href'] for item in soup.select('.list-chapter li a')]

    novel_image = "https://novelfull.com" + soup.select_one('.book img')['src']
    print(f"Found novel image: {novel_image}")

    return novel_title, novel_image, chapter_titles, chapter_urls

def preprocess(text):
    replacements = [
        (r'(?<!\w)\.(\w)\.', r'\1'),
        (r'(?<=\w)\.(\w)', r'\1'),
        (r'(\w)\.(?=\w)', r'\1'),
        (r'\+', ' plus'),
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    return text

# Example usage (Run inside an event loop):
if __name__ == "__main__":
    startTime = datetime.now()
    print("Starting refresh_database task")
    asyncio.run(refresh_database("martial peak"))
    print(f"Time taken: {datetime.now() - startTime}")
