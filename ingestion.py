import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
load_dotenv()

# Set up SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
)
# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vector_store = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map  = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 3) -> List[List[str]]:
  """Split URLs into chunks of specified sizes """
  chunks = []
  for i in range(0, len(urls), chunk_size):
    chunk = urls[i:i + chunk_size]
    chunks.append(chunk)
  return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, any]]:
  """Extract documents from batch of urls"""
  try:
    log_info(f"Processing batch {batch_num} with {len(urls)} URLs", Colors.BLUE)
    docs = await tavily_extract.ainvoke(input= {"urls": urls})
    results = docs.get('results', [])
    log_success(f"Batch {batch_num} completed - extracted {len(results)} documents")
    return docs
  except Exception as e:
    log_error(f"Batch {batch_num} failed: {e}")
    return []




async def index_document_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"VectorStorage Indexing: Preparing to add {len(documents)} to vector store",
        Colors.DARK_CYAN
    )
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"VectorStorage Indexing: Split documents into {len(batches)} batch(es) of {batch_size}.",
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            vector_store.add_documents(batch)
            log_success(f"VectorStorage: Successfully indexed batch {batch_num} with {len(batch)} documents.")
        except Exception as e:
            log_error(f"VectorStorage Indexing: Failed to index batch {batch_num}: {e}")
            return False
        return True

    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)
    if successful == len(batches):
        log_success(
            f"VectorStorage Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStorage Indexing: Processed {successful}/{len(batches)} batches successfully."
        )

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"Starting extraction for {len(url_batches)} batch(es) of URLs.",
        Colors.DARK_CYAN
    )
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: A batch failed with exception: {result}")
            failed_batches += 1
        else:
            pages = result["results"] if isinstance(result, dict) and "results" in result else result
            for extracted_page in pages:
                documents = Document(
                    page_content = extracted_page['raw_content'],
                    metadata = {"source": extracted_page['url']}
                )
                all_pages.append(documents)

    log_success(
        f"TavilyExtract: Extraction completed with {len(all_pages)} document(s) extracted. "
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batch(es) failed during extraction.")
    return all_pages

async def document_helper():
    """document_helper async function to orchestrate this process."""
    log_header("DOCUMENTATION INGESTION PIPELINE FOR DOCUMENT HELPER")
    log_info(
        "TavilyCrawl: Starting to crawl documentation from https://python.langchain.com",
        Colors.PURPLE
    )
    # Crawl the documentation site
    site_map = tavily_crawl.invoke("https://python.langchain.com")
    log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URL(s) from documentation site.")
    # Split the urls into batches of 20
    urls = [item["url"] for item in site_map["results"]]
    url_batches = chunk_urls(urls, 20)
    log_info(
        f"URL processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batch(es) of 20.",
        Colors.BLUE
    )
    all_docs = await async_extract(url_batches)
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"TextSplitter: Processing {len(all_docs)} document(s) with 4000 chunk size and 200 overlap.",
        Colors.YELLOW
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(split_docs)} chunk(s) from {len(all_docs)} document(s)."
    )

    await index_document_async(split_docs, batch_size=500)


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "TavilyCrawl: Starting to crawl documentation from https://python.langchain.com",
        Colors.PURPLE
    )
    # Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 1,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })

    all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in res['results']]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URL(s) from documentation site."
    )

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(document_helper())