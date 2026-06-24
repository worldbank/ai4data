import re
import unicodedata


class TextNormalizer:
    """Utility for cleaning and normalizing text for GLiNER2 extraction.

    Optimized for pymupdf4llm markdown outputs.
    """

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Apply NFKC normalization for consistent character representation."""
        if not text:
            return ""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def normalize_full(text: str, markdown_aware: bool = True) -> str:
        """Full normalization strategy: Unicode + Hyphenation + Paragraph Line Joining.

        Args:
            text: Raw input text
            markdown_aware: If True, avoid joining lines within tables (|) or headers (#)

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # 1. Unicode NFKC
        text = unicodedata.normalize("NFKC", text)

        # 2. Block-aware Line Joining
        lines = text.split("\n")
        normalized_lines = []
        is_in_table = False

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Table Detection
            if "|" in line:
                is_in_table = True
            elif not line:
                is_in_table = False

            if is_in_table:
                # Inside table: only clean spaces, do NOT join lines
                line = re.sub(r" {2,}", " ", line)
                normalized_lines.append(line)
                i += 1
                continue

            # Header or list item Detection (skip joining)
            if line.startswith(("#", "*", "-", "1.", ">")):
                normalized_lines.append(line)
                i += 1
                continue

            # Normal paragraph logic: join lines
            if line:
                # Find next non-empty line
                next_non_empty_idx = -1
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        next_non_empty_idx = j
                        break

                if next_non_empty_idx != -1:
                    next_line = lines[next_non_empty_idx].strip()
                    # Check if next line is structural (headers, lists, tables, links)
                    is_next_structural = next_line.startswith(
                        ("#", "*", "-", "1.", ">", "|", "[", "**")
                    )
                    # Check if current line ends with sentence-ending punctuation
                    ends_with_sentence_terminator = line.endswith((".", "?", "!"))

                    if not is_next_structural and not ends_with_sentence_terminator:
                        if line.endswith("-"):
                            # Join hyphenated word
                            joined = line[:-1] + next_line
                        else:
                            # Join with space
                            joined = line + " " + next_line
                        lines[next_non_empty_idx] = joined
                        i = next_non_empty_idx
                        continue

            # Default: just clean whitespace and add
            cleaned = re.sub(r" {2,}", " ", line)
            if cleaned or not normalized_lines or normalized_lines[-1]:
                normalized_lines.append(cleaned)
            i += 1

        return "\n".join(normalized_lines).strip()

    @staticmethod
    def normalize_simple(text: str) -> str:
        """Lightweight normalization for short labels/metadata.

        Applies Unicode NFKC and collapses multiple spaces into one.
        Use this for mention names, author names, or geography labels
        to ensure they match the normalized input text without
        undergoing line-joining.
        """
        if not text:
            return ""
        s = unicodedata.normalize("NFKC", text)
        # Collapse excessive whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def to_ascii(text: str) -> str:
        """Collapses text to ASCII (strips accents) for robust semantic matching.

        This is an 'Aggressive Normalizer' used to bridge the gap between
        different accent encodings (e.g., 'Cad´Unico' vs 'Cadu').

        Returns:
            Lowercase ASCII string with no accents and single-spaced.
        """
        if not text:
            return ""
        # Normalise to NFD to separate base characters from accents
        s = unicodedata.normalize("NFD", text)
        # Strip all marks (accents) and encode to ASCII
        s = s.encode("ascii", "ignore").decode("utf-8")
        # Lowercase and collapse whitespace for matching
        return re.sub(r"\s+", " ", s).strip().lower()
