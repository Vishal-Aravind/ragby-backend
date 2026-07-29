# sources/gsheets.py

import uuid
import sentry_sdk
import requests
import pandas as pd
from qdrant_client import models

# FIX: removed unused RecursiveCharacterTextSplitter import


def get_sheet_names(sheet_id: str, range_name: str):
    if not range_name or range_name.strip().lower() in ("", "all"):
        return None, []
    requested = [r.strip() for r in range_name.split(",") if r.strip()]
    return requested, []


def fetch_tab(sheet_id: str, tab_name: str):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_name}"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Could not fetch tab '{tab_name}': {e}")
        return None


def get_all_tab_names(sheet_id: str):
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/feed/worksheets?alt=json"
        res = requests.get(url)
        data = res.json()
        entries = data.get("feed", {}).get("entry", [])
        return [e["title"]["$t"] for e in entries]
    except Exception as e:
        sentry_sdk.capture_exception(e)
        print(f"Could not fetch tab list: {e}")
        return None


def sync_sheet(sheet_id: str, range_name: str, project_id: str, source_id: str, qdrant, embeddings, collection: str):
    requested_tabs, _ = get_sheet_names(sheet_id, range_name)

    if requested_tabs is None:
        all_tabs = get_all_tab_names(sheet_id)
        tabs_to_read = all_tabs if all_tabs else [None]
    else:
        tabs_to_read = requested_tabs

    qdrant.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id)
            )]
        )
    )

    all_chunks = []
    all_metas = []
    skipped = []
    synced = []

    for tab in tabs_to_read:
        if tab is None:
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
            try:
                df = pd.read_csv(url)
                tab_label = "default"
            except Exception as e:
                sentry_sdk.capture_exception(e)
                print(f"Could not fetch default tab: {e}")
                continue
        else:
            df = fetch_tab(sheet_id, tab)
            if df is None or df.empty:
                skipped.append(tab)
                continue
            tab_label = tab

        for _, row in df.iterrows():
            # FIX: filter out nan and empty values
            text = ", ".join(
                f"{col}: {row[col]}" for col in df.columns
                if str(row[col]).strip() and str(row[col]) != "nan"
            )
            if not text.strip():
                continue
            all_chunks.append(text)
            all_metas.append({
                "project_id": project_id,
                "source_id": source_id,
                "source_type": "gsheets",
                "sheet_tab": tab_label,
                "text": text,
            })

        synced.append(tab_label)

    if not all_chunks:
        print("No data found in any sheet tab.")
        return {"synced_tabs": synced, "skipped_tabs": skipped}

    vectors = embeddings.embed_documents(all_chunks)

    qdrant.upload_points(
        collection_name=collection,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=v,
                payload=m
            ) for v, m in zip(vectors, all_metas)
        ]
    )

    print(f"Synced {len(all_chunks)} rows from tabs: {synced}, skipped: {skipped}")
    return {"synced_tabs": synced, "skipped_tabs": skipped}