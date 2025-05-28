import aiohttp
from bs4 import BeautifulSoup
import asyncio
import re
from fake_headers import Headers
from logger_config import setup_logger
from utils import get_db_connection

logger = setup_logger("scraper")
header = Headers(browser="chrome", os="win", headers=True)


async def refresh_database():
    headers = header.generate()
    keyword = input("Please enter a keyword to search for novels: ")
    while not keyword:
        logger.error("No keyword provided. Trying again.")
        keyword = input("Please enter a keyword to search for novels: ")
    logger.info(f"Keyword provided: {keyword}")
    while True:
        logger.info(f"Searching for keyword: {keyword}")
        search_url = f"https://novelfull.com/search?keyword={keyword}"
        async with aiohttp.ClientSession(headers=headers) as session:
            search_soup = await fetch_html(session, search_url)
        try:
            novel_title = search_soup.select_one(".truyen-title a").text
        except AttributeError:
            logger.error("No novel title found. Trying again.")
            keyword = input("Please enter a new keyword: ")
            while not keyword:
                logger.error("No keyword provided. Trying again.")
                keyword = input("Please enter a new keyword: ")
            logger.info(f"New keyword provided: {keyword}")
            continue
        logger.info(f"Found novel title: {novel_title}")
        logger.info(f"Getting user confirmation for the novel title: {novel_title}")
        user_response = input(f"Is this the correct novel title? **{novel_title}** (y/n): ")
        if user_response.lower() == "y":
            logger.info(f"User confirmed the novel title: {novel_title}")
            break
        else:
            logger.info(f"User did not confirm the novel title: {novel_title}")
            logger.info("Prompting user for a new keyword.")
            keyword = input("Please enter a new keyword: ")
            while not keyword:
                logger.error("No keyword provided. Trying again.")
                keyword = input("Please enter a new keyword: ")
            logger.info(f"New keyword provided: {keyword}")
            continue
    logger.info(f"Final novel title: {novel_title}")
    logger.info(f"Fetching chapter URLs for {novel_title}")

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        novel_image, chapter_titles, chapter_urls = await get_urls(session, search_soup)

    logger.info(f"Retrieved novel title: {novel_title}, with {len(chapter_titles)} chapters")

    conn = get_db_connection()
    cursor = conn.cursor()

    while True:
        try:
            urls_to_scrape = []
            urls_to_update = []

            # for later I can check directly in the database the list of chapter_urls then fetch them all at once
            # instead of checking one by one using a for loop
            logger.info("Checking database for existing chapters")
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

            logger.info(
                f"URLs to scrape: {len(urls_to_scrape)}, URLs to update: {len(urls_to_update)}"
            )

            headers = header.generate()
            connector = aiohttp.TCPConnector(limit=20)
            async with aiohttp.ClientSession(
                headers=headers, connector=connector
            ) as session:
                await chapter_to_db(
                    session,
                    novel_title,
                    novel_image,
                    urls_to_scrape,
                    urls_to_update,
                    cursor,
                    conn,
                )
                break
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            continue

    cursor.close()
    conn.close()
    logger.info("Database refresh completed")
    return novel_title


async def chapter_to_db(
    session, novel_title, novel_image, urls_to_scrape, urls_to_update, cursor, conn
):
    logger.info(f"Processing novel: {novel_title}")
    cursor.execute("SELECT * FROM novels WHERE novel_title = %s", (novel_title,))
    result = cursor.fetchone()

    if result:
        novel_id = result[0]
    else:
        cursor.execute(
            "INSERT INTO novels (novel_title, novel_image) VALUES (%s, %s) RETURNING id",
            (novel_title, novel_image),
        )
        novel_id = cursor.fetchone()[0]
        conn.commit()
    logger.info(f"Novel ID: {novel_id}")

    chapter_urls = [url[0] for url in urls_to_scrape]
    chapter_titles = [url[1] for url in urls_to_scrape]

    tasks = [
        get_page_content(session, f"https://novelfull.com{chapter_url}")
        for chapter_url in chapter_urls
    ]
    chapter_contents = await asyncio.gather(*tasks)
    chapter_contents = [clean_text(text) for text in chapter_contents]

    for chapter_title, chapter_url, chapter_content in zip(
        chapter_titles, chapter_urls, chapter_contents
    ):
        chapter_number = re.search(r"\d+", chapter_title)
        chapter_number = int(chapter_number.group())

        cursor.execute(
            "INSERT INTO chapters (novel_id, chapter_number, chapter_title, chapter_url, chapter_content) VALUES (%s, %s, %s, %s, %s)",
            (novel_id, chapter_number, chapter_title, chapter_url, chapter_content),
        )
        conn.commit()
        logger.info(
            f"Inserted chapter {chapter_title} into database with number {chapter_number}"
        )

    chapter_urls = [url[0] for url in urls_to_update]
    chapter_titles = [url[1] for url in urls_to_update]

    tasks = [
        get_page_content(session, f"https://novelfull.com{chapter_url}")
        for chapter_url in chapter_urls
    ]
    chapter_contents = await asyncio.gather(*tasks)
    chapter_contents = [clean_text(text) for text in chapter_contents]

    for chapter_title, chapter_content in zip(chapter_titles, chapter_contents):
        chapter_number = re.search(r"\d+", chapter_title)
        chapter_number = int(chapter_number.group())
        cursor.execute(
            "UPDATE chapters SET chapter_content = %s WHERE chapter_number = %s;",
            (chapter_content, chapter_number),
        )
        conn.commit()
        logger.info(
            f"Updated chapter {chapter_title} in database with number {chapter_number}"
        )

    return


async def fetch_html(session, url):
    logger.info(f"Fetching HTML for URL: {url}")
    async with session.get(url) as response:
        html = await response.text()
        return BeautifulSoup(html, "html.parser")


async def get_page_content(session, url):
    soup = await fetch_html(session, url)
    text = "\n".join(
        [
            item.text
            for item in soup.select("p")
            if item.text.strip()
            and "translator" not in item.text.lower()
            and "copyright" not in item.text.lower()
        ]
    )
    return text


async def get_urls(session, search_soup):

    chapter_list_url = (
        f"https://novelfull.com{search_soup.select_one('.truyen-title a')['href']}"
    )
    chapter_list_soup = await fetch_html(session, chapter_list_url)

    last_url = (
        f'https://novelfull.com{chapter_list_soup.select_one("li.last a")["href"]}'
    )
    last_page = int(last_url.split("=")[-1])
    url_part1 = last_url.split("page=")[0] + "page="

    tasks = [
        fetch_html(session, f"{url_part1}{page_num}")
        for page_num in range(1, last_page + 1)
    ]
    page_soups = await asyncio.gather(*tasks)

    chapter_titles = []
    chapter_urls = []

    for soup in page_soups:
        chapter_titles += [item.text for item in soup.select("#list-chapter .row li a")]
        chapter_urls += [item["href"] for item in soup.select(".list-chapter li a")]

    novel_image = "https://novelfull.com" + soup.select_one(".book img")["src"]
    logger.info(f"Found novel image: {novel_image}")

    return novel_image, chapter_titles, chapter_urls


def clean_text(text):
    replacements = [
        (r"(?<!\w)\.(\w)\.", r"\1"),
        (r"(?<=\w)\.(\w)", r"\1"),
        (r"(\w)\.(?=\w)", r"\1"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    return text


if __name__ == "__main__":
    asyncio.run(refresh_database())