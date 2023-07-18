import requests
from bs4 import BeautifulSoup
from typing import Tuple, Dict, Any
from llama_index import Document


def page_ingest(url) -> Tuple[str, Dict[str, Any]]:

    print("url", url)
    label = ''

    # Fetch the content from url
    response = requests.get(url)
    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.text, 'html.parser')

    # Initialize an empty string to hold text
    text = ''
    # Initialize an empty dictionary to hold code
    code_blocks = {}

    # Extract all text not contained in a script or style element
    text_elements = soup.findAll(text=True)
    for element in text_elements:
        if element.parent.name not in ['script', 'style', 'a']:
            text += element.strip()
    print(len(text), url)

    document = Document(text=text, extra_info={'source': url})
    print(document)
    return document


def ingest_main(list_urls):
    list_of_docs = []
    for url in list_urls:
        page = page_ingest(url)
        list_of_docs.append(page)
    return list_of_docs


__all__ = ['ingest_main']
