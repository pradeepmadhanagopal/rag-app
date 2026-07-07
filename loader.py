from pypdf import PdfReader

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([page.extract_text() for page in reader.pages])

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    text = load_pdf("docs/4a_Convolution.pdf")
    print(f"Extracted {len(text)} characters from PDF")
    chunks = chunk_text(text)
    
    print(f"Split into {len(chunks)} chunks\n")
    print("--- Chunk 0 ---")
    print(chunks[0])
    print("--- Chunk 1 ---")
    print(chunks[1])
    