import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


def load_file(path: str) -> list[Document]:
    """
    Loads a document and attaches source metadata.
    Supports: PDF, DOCX, CSV, XLSX, TXT
    """
    ext = path.split(".")[-1].lower()
    filename = os.path.basename(path)
    docs = []

    # ---------- PDF ----------
    if ext == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
            if "page" not in d.metadata:
                d.metadata["page"] = 0
            else:
                # PyPDFLoader uses 0-based index; make it 1-based for display
                d.metadata["page"] = d.metadata["page"] + 1
        return docs

    # ---------- DOCX ----------
    if ext == "docx":
        loader = Docx2txtLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = filename
            d.metadata["page"] = ""
        return docs

    # ---------- CSV ----------
    if ext == "csv":
        df = pd.read_csv(path)
        # Split into smaller logical chunks by rows (100 rows each)
        chunk_size = 100
        chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk.to_string(index=False),
                metadata={"source": filename, "page": f"rows {idx*chunk_size+1}-{min((idx+1)*chunk_size, len(df))}"}
            )
            docs.append(doc)
        return docs

    # ---------- Excel ----------
    if ext in ("xlsx", "xls"):
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            chunk_size = 100
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=f"Sheet: {sheet}\n{chunk.to_string(index=False)}",
                    metadata={"source": filename, "page": f"Sheet:{sheet} rows {idx*chunk_size+1}-{min((idx+1)*chunk_size, len(df))}"}
                )
                docs.append(doc)
        return docs

    # ---------- TXT ----------
    if ext == "txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        doc = Document(
            page_content=content,
            metadata={"source": filename, "page": ""}
        )
        return [doc]

    # ---------- Unsupported ----------
    return []
