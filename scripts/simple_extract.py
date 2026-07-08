#!/usr/bin/env python
"""Simple dataset extraction script using ai4data.data_use."""

import os
import sys
import json

# Add repository src/ to Python path if running locally
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

from ai4data import extract_from_text


def main():
    text = (
        "In this paper, we explore how agricultural productivity shapes household consumption patterns in Sub-Saharan Africa. "
        "We conduct our main analysis using microdata from the 2018 Nigeria General Household Survey (GHS), "
        "which was conducted by the National Bureau of Statistics (NBS) in collaboration with the World Bank. "
        "The GHS contains detailed survey responses across 36 states in Nigeria. "
        "To complement these findings and check macroeconomic trends, we integrate indicators from the "
        "World Development Indicators (WDI) compiled by the World Bank database covering the years 2010 to 2020. "
        "Finally, as a preliminary reference, we mention population counts from the 2006 Population and "
        "Housing Census of Nigeria, published by the National Population Commission."
    )

    print("--- Input Text ---")
    print(text)
    print("\nExtracting dataset mentions...")

    # Run extraction using the datause adapter
    result = extract_from_text(
        text,
        include_confidence=True,
        adapter_id="ai4data/datause-extraction",
    )

    print("\n--- Extracted Datasets ---")
    datasets = result.get("datasets", [])
    if not datasets:
        print("No datasets found.")
    for ds in datasets:
        print(f"Mention:       {ds['mention_name']['text']}")
        print(f"  Confidence:  {ds['mention_name']['confidence']:.3f}")
        print(f"  Acronym:     {ds['acronym']['text'] or 'None'}")
        print(f"  Producer:    {ds['producer']['text'] or 'None'}")
        print(f"  Year:        {ds['reference_year']['text'] or 'None'}")
        print(f"  Typology:    {ds['typology_tag']['text']} (confidence: {ds['typology_tag']['confidence']:.3f})")
        print(f"  Usage:       {ds['usage_context']['text']} (confidence: {ds['usage_context']['confidence']:.3f})")
        print("-" * 50)


if __name__ == "__main__":
    main()
