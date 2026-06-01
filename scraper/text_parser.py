import os
import fitz  # PyMuPDF
import re
from tqdm import tqdm
import json

PDF_DIR = "corpus_pdfs"
OUTPUT_FILE = "eval/parsed_texts.jsonl"

def clean_text(text):
    """Removes messy PDF formatting like mid-word hyphens and weird spacing."""
    # Remove hyphenation at the end of lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace single newlines with spaces (keeps paragraph double-newlines)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_pdf(pdf_path):
    """Extracts text from a single PDF."""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            full_text.append(text)
        
        raw_text = "\n\n".join(full_text)
        return clean_text(raw_text)
    except Exception as e:
        print(f"Error parsing {pdf_path}: {e}")
        return None

def main():
    if not os.path.exists(PDF_DIR):
        print(f"Directory {PDF_DIR} not found.")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs to parse.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for filename in tqdm(pdf_files, desc="Parsing PDFs"):
            arxiv_id = filename.replace(".pdf", "")
            filepath = os.path.join(PDF_DIR, filename)
            
            parsed_text = parse_pdf(filepath)
            
            if parsed_text:
                # Save as JSONL so we can process it line-by-line later without blowing up RAM
                record = {
                    "arxiv_id": arxiv_id,
                    "text": parsed_text
                }
                out_f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()