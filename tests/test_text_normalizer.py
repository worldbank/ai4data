"""Tests for TextNormalizer."""

from ai4data.data_use.utils.text_normalizer import TextNormalizer


def test_normalize_unicode():
    """Test unicode normalization."""
    assert TextNormalizer.normalize_unicode("fi\ufb01") == "fifi"


def test_normalize_simple():
    """Test simple whitespace normalization."""
    assert TextNormalizer.normalize_simple("  Some   noisy    spaces  ") == "Some noisy spaces"


def test_normalize_full_reconstruction():
    """Test paragraph line joining and reconstruction on soft-broken text."""
    noisy_text = (
        "A second contribution of this paper is to show the importance of decomposing the effects with\n"
        "respect to distance from the mines. Given the spatial heterogeneity of the results, we explore\n"
        "the effects in an individual-level, difference-in-differences analysis by using spatial lag models\n"
        "to allow for nonlinear effects with distance from mine. We also allow for spillovers across\n"
        "districts, in a district-level analysis. We use two complementary geocoded household data sets\n"
        "to analyze outcomes in \n"
        "Ghana : the \n"
        "Demographic and Health Survey\n"
        " (\n"
        "DHS \n"
        ") and the \n"
        "Ghana\n"
        "Living Standard Survey (GLSS) \n"
        ", which provide information on a wide range of welfare\n"
        "outcomes."
    )

    expected_normalized = (
        "A second contribution of this paper is to show the importance of decomposing the effects with "
        "respect to distance from the mines. Given the spatial heterogeneity of the results, we explore "
        "the effects in an individual-level, difference-in-differences analysis by using spatial lag models "
        "to allow for nonlinear effects with distance from mine. We also allow for spillovers across "
        "districts, in a district-level analysis. We use two complementary geocoded household data sets "
        "to analyze outcomes in Ghana : the Demographic and Health Survey ( DHS ) and the Ghana "
        "Living Standard Survey (GLSS) , which provide information on a wide range of welfare outcomes."
    )

    normalized = TextNormalizer.normalize_full(noisy_text)
    assert normalized == expected_normalized


def test_normalize_full_hyphenation():
    """Test that hyphenation joining still works."""
    hyphen_text = "This is a multi-\npage document with standard line-breaks."
    expected = "This is a multipage document with standard line-breaks."
    assert TextNormalizer.normalize_full(hyphen_text) == expected


def test_normalize_full_preserves_markdown():
    """Test that markdown structural elements are preserved during joining."""
    md_text = (
        "# Title\n\n"
        "First line of paragraph\n"
        "second line of paragraph.\n\n"
        "* List item 1\n"
        "* List item 2\n\n"
        "| Col 1 | Col 2 |\n"
        "|---|---|\n"
        "| Val 1 | Val 2 |"
    )

    expected = (
        "# Title\n\n"
        "First line of paragraph second line of paragraph.\n\n"
        "* List item 1\n"
        "* List item 2\n\n"
        "| Col 1 | Col 2 |\n"
        "|---|---|\n"
        "| Val 1 | Val 2 |"
    )

    assert TextNormalizer.normalize_full(md_text) == expected
