

def load_pdf(
    fname_or_url: str, 
    n_pages: int = 1,
    parser: str = "pymupdf"
) -> List[Dict[str, Union[str, List[int]]]]:
    """
    Loads a PDF using the selected parser.
    Parameters:
        fname_or_url: Local file path or URL to PDF.
        n_pages: Number of pages per chunk (for pymupdf).
        parser: "pymupdf" or "mistral".
    Returns:
        List of dicts with 'text' and 'pages' keys.
    """

    def _open_pymupdf(path: str) -> List[Dict[str, Union[str, List[int]]]]:
        import fitz  # pymupdf
        doc = fitz.open(path)
        total = len(doc)
        for start in range(0, total, n_pages):
            end = min(start + n_pages, total)
            texts = [doc[i].get_text() for i in range(start, end)]
            yield {"text": "\n\n".join(texts), "pages": list(range(start, end))}
        doc.close()

    def _open_mistral(path_or_url: str) -> List[Dict[str, Union[str, List[int]]]]:
        ocr_output = mistral_ocr_func(path_or_url)
        return mistral_ocr_to_pymupdf_format(ocr_output)

    # URL handling: always download to temp file for pymupdf
    if parser == "pymupdf":
        if fname_or_url.startswith("http://") or fname_or_url.startswith("https://"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                r = requests.get(fname_or_url, stream=True)
                r.raise_for_status()
                for chunk in r.iter_content(8192):
                    tmp.write(chunk)
                tmp.flush()
                tmp_path = tmp.name
            results = list(_open_pymupdf(tmp_path))
            os.unlink(tmp_path)
            return results
        else:
            return list(_open_pymupdf(fname_or_url))
    elif parser == "mistral":
        # mistral can handle both local files and URLs
        return _open_mistral(fname_or_url)
    else:
        raise ValueError(f"Unknown parser: {parser}")

def load_json_data(filepath: str):
    """
    Loads a JSON or JSONL file from the given path.
    Returns a dict or list (for JSON) or list of dicts (for JSONL).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.strip().startswith('{') and not first_line.strip().endswith('['):
            # Try JSON Lines: each line is a dict
            try:
                data = [json.loads(line) for line in f if line.strip()]
                if len(data) == 1:
                    f.seek(0)
                    data = json.load(f)
                return data
            except Exception:
                f.seek(0)
                return json.load(f)
        else:
            return json.load(f)