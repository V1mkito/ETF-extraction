#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM-powered attribute extraction for ETF PDFs.

Workflow:
1. Input: per-ETF text JSON files (output of your PDF text extraction script).
2. For each JSON:
   - Build SecuritiesInformation.source_documents from the text JSON.
   - Call an OpenAI model to extract numeric + categorical attributes
     into the FNDA-style "extracted_attributes" schema.
3. Output: one JSON per ETF, following the FNDA example structure.

Run example:
    python 02_llm_attribute_extraction.py \
        --input-dir ./text_json \
        --output-dir ./etf_attributes \
        --model gpt-4o-mini
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

# Make sure OPENAI_API_KEY is set in your environment before running:
#   PowerShell:  $env:OPENAI_API_KEY = "sk-xxxx..."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior financial data analyst specializing in ETFs.

Your task:
Given the full text of disclosure PDFs for ONE ETF (fact sheet, prospectus,
annual/quarterly reports, portfolio management reports, etc.), extract
structured attributes into the following JSON schema, and return ONLY JSON.

You must output a JSON object with ONE top-level key:

  {
    "SecuritiesInformation": {
      "security_type": "etf",
      "security_name": string,
      "security_ticker": string,
      "source_documents": [...],
      "extracted_attributes": {
        "Fundamentals": {
          "ValuationMetrics": {
            "PriceEarningsRatio": {
              "value": number,
              "unit": "ratio",
              "as_of_date": string | null,
              "source_document_id": string,
              "source_page": integer
            },
            "PriceBookRatio": { ... same shape ... },
            "ReturnOnEquity": { ... }
          },
          "IncomeMetrics": {
            "DividendYield": { ... },
            "DistributionFrequency": { ... }
          }
        },
        "PerformanceRiskAdjusted": {
          "Returns": {
            "OneYearNAVReturn": { ... },
            "ThreeYearNAVReturn": { ... },
            "FiveYearNAVReturn": { ... },
            "TenYearNAVReturn": { ... optional ... },
            "SinceInceptionNAVReturn": { ... optional ... }
          },
          "RiskMetrics": {
            "SharpeRatio": { ... },
            "StandardDeviation": { ... },
            "Beta": { ... }
          }
        },
        "LiquidityAndExpenses": {
          "AUM": {
            "TotalNetAssets": { ... }
          },
          "Trading": {
            "AverageDailyVolume": { ... },
            "PortfolioTurnover": { ... }
          },
          "Costs": {
            "TotalExpenseRatio": { ... }
          }
        },
        "BrokerageSpecificAttributes": {
          "Provider": {
            "ProviderName": {
              "value": string,
              "source_document_id": string,
              "source_page": integer
            }
            // Additional broker-specific text attributes are allowed here,
            // but they MUST follow the same object structure:
            //   { "value": string | number,
            //     "unit": string | null,
            //     "as_of_date": string | null,
            //     "source_document_id": string,
            //     "source_page": integer,
            //     "description": string | null }
          }
        }
      }
    }
  }

Important rules:
- Use JSON mode: your reply MUST be valid JSON and nothing else.
- If an attribute is not clearly present in the text, simply OMIT that key.
  Do NOT invent values.
- "as_of_date" should be a string (e.g., "2025-09-30") when available,
  otherwise null or omitted.
- "unit" should be a short code like "percentage", "ratio", "dollars",
  "shares_3m_avg", etc., when obvious.
- "source_document_id" must match one of the document_id values you see in the input.
- "source_page" is the page_number within that document where the value appears.
- For security_name and security_ticker, use the clearest value from the documents
  (usually on the fact sheet or prospectus cover). If still unclear, you may use
  the provided ticker hint from the user context.
"""

USER_PROMPT_TEMPLATE = """
You are given parsed PDF text for one ETF.

Ticker hint: {ticker_hint}

The content is provided as a JSON array of documents.
Each document has:
  - document_id (string)
  - file_name (string)
  - document_type (string)
  - pages: list of objects with "page_number" and "page_text".

Use ONLY this content to populate the "extracted_attributes" section
of the schema described in the system prompt.

Input documents JSON:
{documents_json}

Output MUST be a single valid JSON object only.
No markdown, no code fences, no explanations.
If a field is missing, use null.
"""


# ---------------------------------------------------------------------------
# Helpers for reading input JSON (from your text-extraction step)
# ---------------------------------------------------------------------------

def load_text_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_source_documents(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert your previous text-extraction JSON format into the
    SecuritiesInformation.source_documents format expected by the FNDA schema.

    Expected raw format (simplified):

      {
        "project": {...},
        "process": {...},
        "documents": [
          {
            "metadata": {
              "file_metadata": {
                "file_ID": {"value": "..."},
                "file_name": {"value": "..."},
                "section_or_volume_title": {"value": "..."},
                "main_document": {"value": "..."},
                "total_pages": {"value": "..."},
                "file_provider": {"value": "..."}
              }
            },
            "pages": [
              { "page number": 1, "page text": "..." },
              ...
            ]
          },
          ...
        ]
      }
    """
    docs_out: List[Dict[str, Any]] = []

    for i, doc in enumerate(raw.get("documents", []), start=1):
        meta = (
            doc.get("metadata", {})
               .get("file_metadata", {})
        )

        file_id = (meta.get("file_ID", {}) or {}).get("value") or f"doc_{i}"
        file_name = (meta.get("file_name", {}) or {}).get("value") or ""
        section_title = (meta.get("section_or_volume_title", {}) or {}).get("value") or ""
        document_type = meta.get("document_type", {}).get("value") if "document_type" in meta else ""
        total_pages_val = (meta.get("total_pages", {}) or {}).get("value") or ""
        file_provider = (meta.get("file_provider", {}) or {}).get("value") or ""

        # Convert pages
        pages_raw = doc.get("pages", [])
        pages_out = []
        for p in pages_raw:
            # Be defensive about key names
            page_number = p.get("page number") or p.get("page_number") or p.get("page") or 0
            page_text = p.get("page text") or p.get("page_text") or ""
            pages_out.append(
                {
                    "page_number": page_number,
                    "page_text": page_text,
                }
            )

        # Try to infer document_date if present in metadata (optional)
        document_date = (meta.get("document_date", {}) or {}).get("value") or None

        docs_out.append(
            {
                "document_id": file_id,
                "file_name": file_name,
                "section_or_volume_title": section_title,
                "document_type": document_type,
                "total_pages": int(total_pages_val) if str(total_pages_val).isdigit() else total_pages_val,
                "file_provider": file_provider,
                "document_date": document_date,
                "pages": pages_out,
            }
        )

    return docs_out


def guess_security_name(raw: Dict[str, Any]) -> str:
    """
    Try to guess the ETF name from the project title or file metadata.
    If not found, return an empty string and let the LLM refine it.
    """
    project = raw.get("project", {})
    title_val = (project.get("project_title", {}) or {}).get("value")
    if isinstance(title_val, str) and title_val.strip():
        return title_val.strip()

    # Fallback: use first file_name without extension
    documents = raw.get("documents", [])
    if documents:
        meta = documents[0].get("metadata", {}).get("file_metadata", {})
        fname = (meta.get("file_name", {}) or {}).get("value") or ""
        return os.path.splitext(fname)[0] if fname else ""

    return ""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm_for_attributes(
    ticker: str,
    source_documents: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Call OpenAI Responses API to extract attributes into the FNDA-style
    "SecuritiesInformation" JSON structure.

    We ask the model to output the FULL "SecuritiesInformation" object,
    not just extracted_attributes, so it can refine security_name/ticker
    if needed.
    """

    # Truncate long text to avoid hitting max token limits
    # (very crude but safe for many cases)
    documents_json_str = json.dumps(source_documents, ensure_ascii=False)
    max_chars = 120_000
    if len(documents_json_str) > max_chars:
        documents_json_str = documents_json_str[:max_chars]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        ticker_hint=ticker,
        documents_json=documents_json_str,
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=4000,
    )

    # responses.create returns an object; output_text is the full JSON string
    json_text = response.output_text

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        # Fallback: wrap into a minimal object so the script doesn't crash
        print(f"[WARN] Failed to parse JSON for {ticker}: {exc}")
        parsed = {
            "SecuritiesInformation": {
                "security_type": "etf",
                "security_name": "",
                "security_ticker": ticker,
                "source_documents": source_documents,
                "extracted_attributes": {},
            }
        }

    # Ensure we always inject our source_documents (LLM might omit or change them)
    if "SecuritiesInformation" not in parsed:
        parsed["SecuritiesInformation"] = {}

    si = parsed["SecuritiesInformation"]
    si.setdefault("security_type", "etf")
    si.setdefault("security_ticker", ticker)
    si["source_documents"] = source_documents

    return parsed


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def process_file(path: Path, output_dir: Path, model: str) -> None:
    print(f"[INFO] Processing {path.name} ...")
    raw = load_text_json(path)

    # Ticker from filename, e.g. FBCG.json -> FBCG
    ticker = path.stem.upper()

    source_docs = build_source_documents(raw)
    result = call_llm_for_attributes(ticker=ticker, source_documents=source_docs, model=model)

    out_path = output_dir / path.name  # keep same name, different folder
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved attributes JSON to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-based attribute extraction for ETF text JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing per-ETF text JSON files (output from PDF text extraction).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write FNDA-style attribute JSON files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name to use (default: gpt-4o-mini).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    json_files = sorted(p for p in input_dir.glob("*.json") if p.is_file())
    if not json_files:
        print(f"[WARN] No .json files found in {input_dir}")
        return

    print(f"[INFO] Found {len(json_files)} JSON files in {input_dir}")

    for path in json_files:
        process_file(path, output_dir, model=args.model)

    print("[DONE] All files processed.")


if __name__ == "__main__":
    main()
