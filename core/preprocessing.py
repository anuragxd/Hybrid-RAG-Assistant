import trafilatura
from utils.logger import get_logger

logger = get_logger(__name__)

def clean_html(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        logger.warning(f"Failed to fetch {url}")
        return ""
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    return text if text else ""