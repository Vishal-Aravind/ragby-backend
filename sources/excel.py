# sources/excel.py

import uuid
import io
import requests
import pandas as pd
from qdrant_client import models


def fetch_excel_from_url(url: str) -> bytes:
    """
    Fetch Excel bytes from a URL, handling OneDrive/SharePoint sharing links.
    """
    session = requests.Session()
    res = session.get(url, allow_redirects=True, timeout=30)
    resolved_url = res.url

    content_type = res.headers.get("Content-Type", "")
    if "html" in content_type:
        if "onedrive.live.com" in resolved_url:
            resolved_url = resolved_url.replace("redir?", "download?")
            resolved_url = resolved_url.replace("download=0", "download=1")
            if "download=" not in resolved_url:
                resolved_url += "&download=1"
            res = session.get(resolved_url, allow_redirects=True, timeout=30)
        elif "sharepoint.com" in resolved_url:
            sep = "&" if "?" in resolved_url else "?"
            res = session.get(resolved_url + sep + "download=1", allow_redirects=True, timeout=30)
        else:
            raise ValueError(
                "This link returns a webpage, not an Excel file. "
                "Please use a direct download link.\n\n"
                "In OneDrive: open the file → File → Download → copy the browser URL before it saves."
            )

    content_type = res.headers.get("Content-Type", "")
    if "html" in content_type:
        raise ValueError(
            "Could not get a direct download from this OneDrive link. "
            "Please open the file in OneDrive, click File → Download, "
            "and copy the URL from the browser address bar before the file saves."
        )

    return res.content


def excel_bytes_to_chunks(file_bytes: bytes):
    """
    Convert Excel bytes to list of (sheet_name, text) tuples.
    FIX: removed duplicate definition — keeping the better version with
    engine detection (openpyxl/xlrd) and nan filtering.
    """
    # Try openpyxl first (xlsx), fall back to xlrd (xls)
    for engine in ["openpyxl", "xlrd"]:
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
            results = []
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name).astype(str).fillna("")
                for _, row in df.iterrows():
                    text = ", ".join(
                        f"{col}: {row[col]}" for col in df.columns
                        if str(row[col]).strip() and str(row[col]) != "nan"
                    )
                    if text.strip():
                        results.append((sheet_name, text))
            return results
        except Exception:
            continue

    raise ValueError("Could not read the Excel file. Make sure it is a valid .xlsx or .xls file.")


def sync_excel_url(
    url: str,
    project_id: str,
    source_id: str,
    qdrant,
    embeddings,
    collection: str
):
    """Fetch Excel from a URL, embed and store in Qdrant."""
    file_bytes = fetch_excel_from_url(url)
    return _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection, source_label="excel_online")


def sync_excel_bytes(
    file_bytes: bytes,
    project_id: str,
    source_id: str,
    qdrant,
    embeddings,
    collection: str
):
    """Embed and store local Excel bytes in Qdrant."""
    return _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection, source_label="excel_local")


def _sync_excel_bytes(file_bytes, project_id, source_id, qdrant, embeddings, collection, source_label):
    qdrant.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id)
            )]
        )
    )

    rows = excel_bytes_to_chunks(file_bytes)

    if not rows:
        return {"chunks_indexed": 0}

    chunks = [text for _, text in rows]
    metas = [
        {
            "project_id": project_id,
            "source_id": source_id,
            "source_type": "excel",
            "sheet_tab": sheet,
            "text": text,
        }
        for sheet, text in rows
    ]

    vectors = embeddings.embed_documents(chunks)

    qdrant.upload_points(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload=m
            ) for v, m in zip(vectors, metas)
        ]
    )

    print(f"[{source_label}] Synced {len(chunks)} rows from Excel")
    return {"chunks_indexed": len(chunks)}