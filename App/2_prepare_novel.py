from scraper import refresh_database
from logger_config import setup_logger
from chunker import chunking_novel
from indexer import indexing_novel_chunks_chroma, indexing_novel_chunks_bm25
import asyncio

logger = setup_logger("get_novel")


async def main():
    """
    Main function to refresh the database with the latest novels based on the provided keyword.
    """
    logger.info("Starting Novel preparation.")

    logger.info("Calling Scraper to refresh the Database...")
    novel_name = await refresh_database()
    logger.info("Database refreshed successfully.")

    logger.info("Calling Chunker to chunk the novel...")
    chunking_novel(novel_name)
    logger.info("Novel chunking completed successfully.")

    logger.info("Calling indexer to index the chunks...")
    indexing_novel_chunks_chroma(novel_name)
    indexing_novel_chunks_bm25(novel_name)
    logger.info("Indexer completed successfully.")

    logger.info("Novel preparation completed.")
    return


if __name__ == "__main__":
    asyncio.run(main())