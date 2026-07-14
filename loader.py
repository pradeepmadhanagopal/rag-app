from pypdf import PdfReader
import re

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return clean_text("\n".join(page.extract_text() for page in reader.pages))

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

SINGLE_LETTER_TAIL = re.compile(r"([a-z]{2,}) ([a-z])(?![a-z])") 

KNOWN_FIXES = {
    "angl es": "angles",
    "neuroan atomy": "neuroanatomy",
    "featu res": "features",
}

def clean_text(text: str) -> str: 
    text = SINGLE_LETTER_TAIL.sub(r"\1\2", text)
    for bad, good in KNOWN_FIXES.items():
        text = text.replace(bad, good)
    return text  
        

if __name__ == "__main__":
    text = load_pdf("docs/4a_Convolution.pdf")
    print(f"Extracted {len(text)} characters from PDF")
    chunks = chunk_text(text)
    
    print(f"Split into {len(chunks)} chunks\n")
    print("--- Chunk 0 ---")
    print(chunks[0])
    print("--- Chunk 1 ---")
    print(chunks[1])
    