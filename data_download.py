import os
import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Constants
METADATA_FILE = "C:/Users/Nitin/Desktop/aims-dtu/eval/corpus_metadata_filtered.json"
OUTPUT_DIR = "corpus_pdfs"
MAX_WORKERS = 5 # arXiv might rate-limit if this is too high; 5 is usually safe.

def download_pdf(paper):
    """Downloads a single PDF if it doesn't already exist."""
    arxiv_id = paper.get('arxiv_id')
    pdf_url = paper.get('pdf_url')
    
    if not arxiv_id or not pdf_url:
        return False
        
    # Ensure the URL is https and ends with .pdf for a direct download
    if pdf_url.startswith("http://"):
        pdf_url = pdf_url.replace("http://", "https://")
    if not pdf_url.endswith(".pdf"):
        pdf_url += ".pdf"
        
    pdf_path = os.path.join(OUTPUT_DIR, f"{arxiv_id}.pdf")
    
    # Resume capability: skip if already downloaded
    if os.path.exists(pdf_path):
        return True
    
    try:
        # Add a basic User-Agent header to avoid being blocked
        req = urllib.request.Request(
            pdf_url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(pdf_path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except urllib.error.URLError as e:
        print(f"\nFailed to download {arxiv_id}: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error on {arxiv_id}: {e}")
        return False

def main():
    # 1. Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Load the collected metadata
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Could not find {METADATA_FILE}. Run the collection script first.")
        return
        
    with open(METADATA_FILE, "r") as f:
        papers = json.load(f)
        
    print(f"Found {len(papers)} papers in metadata. Starting download...")
    
    # 3. Download in parallel with a progress bar
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the download function to our list of papers
        results = list(tqdm(
            executor.map(download_pdf, papers), 
            total=len(papers), 
            desc="Downloading PDFs"
        ))
        
    success_count = sum(1 for r in results if r is True)
    print(f"\nFinished! Successfully downloaded/verified {success_count} out of {len(papers)} PDFs.")

if __name__ == "__main__":
    main()