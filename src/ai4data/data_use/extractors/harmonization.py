import ast
import glob
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict

import geonamescache
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from .harmonization_adapter import extract_dataset_mentions, get_dedup_folder_structure

### Utility functions


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Parameters
    ----------
    data : any
        Data to save (must be JSON serializable).
    file_path : str
        Path where the JSON file will be saved.
    indent : int, optional
        Indentation level for pretty printing (default: 2).
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


### helpers


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def normalize_acronym_field(x):
    """
    Return a clean list of strings for the 'acronyms' field.
    Handles: None/NaN, list, JSON string, Python-repr string, plain string.
    """
    if x is None or _is_nan(x):
        return []
    if isinstance(x, list):
        return [str(a).strip() for a in x if a is not None and str(a).strip()]
    if isinstance(x, str):
        s = x.strip()
        # Try JSON first: ["DHS", "NVDRS"]
        if s.startswith("[") and s.endswith("]"):
            # JSON attempt
            try:
                return [str(a).strip() for a in json.loads(s)]
            except Exception:
                # Python repr attempt: ['DHS', 'NVDRS']
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, list):
                        return [str(a).strip() for a in v if a is not None and str(a).strip()]
                except Exception:
                    pass
        # Fallback: split on common separators
        parts = re.split(r"[;,|/]", s)
        return [p.strip() for p in parts if p.strip()]
    # Anything else: coerce to single-item list
    return [str(x).strip()]


def pick_plausible_acronym(dataset_name, acronyms):
    """
    Choose a plausible acronym from a list, biased toward:
      1) Exact match to initials of dataset_name
      2) All-caps tokens length 2-12
      3) Otherwise first non-empty candidate
    """
    cands = [a.strip() for a in normalize_acronym_field(acronyms) if a and a.strip()]
    if not cands:
        return None

    # Derive initials from dataset_name
    tokens = re.findall(r"[A-Za-z]+", str(dataset_name) or "")
    initials = "".join(t[0] for t in tokens if t).upper()

    # exact match to initials
    if initials and initials in {c.upper() for c in cands}:
        return initials

    # prefer all-caps length 2-12
    caps = [c for c in cands if c.isupper() and 2 <= len(c) <= 12]
    if caps:
        # if multiple, prefer the one most similar to initials by simple score
        def score(c):
            return sum(1 for a, b in zip(c, initials) if a == b), -abs(len(c) - len(initials))

        return sorted(caps, key=score, reverse=True)[0]

    # fallback: first non-empty
    return cands[0]


# Normalize
def normalize(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt)
    txt = txt.encode("ascii", "ignore").decode("utf-8")
    txt = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# Prebuild regex of all country/demonym forms
def build_country_regex(country_map: dict) -> re.Pattern:
    forms = set()
    for v in country_map.values():
        for f in v:
            forms.add(normalize(f))
    # Sort longest first so multi-word forms like "cote divoire" get matched before "cote"
    pattern = r"\b(" + "|".join(sorted(map(re.escape, forms), key=len, reverse=True)) + r")\b"
    return re.compile(pattern)


# Strip country/demonym
def strip_country(text: str, country_pattern: re.Pattern) -> str:
    clean = normalize(text)
    clean = country_pattern.sub(" ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# Strip years/numbers
def strip_numbers_years(text: str) -> str:
    text = re.sub(r"\b(19|20)\d{2}\b", "", text)  # 4-digit years
    text = re.sub(r"\d+\b", "", text)  # trailing numbers
    text = re.sub(r"\b[ivxlcdm]{1,4}\b", "", text, flags=re.I)  # Roman numerals
    text = re.sub(r"\bof\b\s*$", "", text)  # dangling 'of'
    text = re.sub(r"\s+", " ", text).strip()
    return text


def base_name_norm(raw: str, country_pattern, lemmatizer, stopwords):
    if not isinstance(raw, str):
        return ""
    # Remove countries/demonyms
    txt = strip_country(raw, country_pattern)
    # Remove numbers/years
    txt = strip_numbers_years(txt)
    # Normalize (lowercase + strip accents/punct)
    txt = re.sub(r"\bU\.?\s*S\.?\b", "united states", txt, flags=re.I)
    txt = re.sub(r"\bU\.?\s*K\.?\b", "united kingdom", txt, flags=re.I)
    txt = normalize(txt)

    # Lemmatize + drop stopwords
    tokens = [lemmatizer.lemmatize(tok) for tok in txt.split() if tok not in stopwords]
    return " ".join(tokens)


def build_country_map_with_cities_only(
    user_country_map=None, min_population=100000, fuzzy=True, score_cutoff=90
):
    """
    Build a country_map with:
      - canonical country names
      - cities above min_population
      - user-supplied aliases (merged via fuzzy matching if needed)

    Excludes ISO2/ISO3 codes.
    """
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    cities = gc.get_cities()

    # GeoNames country names (canonical)
    country_map = {}
    for iso2, info in countries.items():
        cname = info["name"].lower()
        aliases = {cname}  # keep only the canonical name
        country_map[cname] = aliases

    # Add cities , I found cities in the names of the datasets i.e Moscow
    for cid, info in cities.items():
        iso2 = info["countrycode"]
        cname = countries[iso2]["name"].lower()
        if int(info.get("population", 0)) >= min_population:
            country_map[cname].add(info["name"].lower())

    # Merge user-supplied aliases
    if user_country_map:
        for key, aliases in user_country_map.items():
            norm_key = key.lower()
            if norm_key in country_map:
                country_map[norm_key].update(a.lower() for a in aliases)
            elif fuzzy:
                result = process.extractOne(
                    norm_key, list(country_map.keys()), score_cutoff=score_cutoff
                )
                if result:
                    match, score, _ = result
                    country_map[match].update(a.lower() for a in aliases)
                else:
                    country_map[norm_key] = set(a.lower() for a in aliases)
            else:
                country_map[norm_key] = set(a.lower() for a in aliases)

    return {k: sorted(v) for k, v in country_map.items()}


# Dataset Clustering Pipeline (Pre-Hierarchy)


def is_acronym(txt: str) -> bool:
    """Return True if the text looks like an acronym (short, uppercase, or a mix of caps and digits)."""
    return len(txt) <= 6 and re.fullmatch(r"[A-Z0-9\-]+", txt or "") is not None


def prefilter(df):
    """Remove entries that are too short or just numbers, and mark which ones are acronyms."""
    df = df.copy()
    df["is_acronym"] = df["base_name_norm"].apply(is_acronym)
    df = df[df["base_name_norm"].str.len() > 2]
    df = df[~df["base_name_norm"].str.isnumeric()]
    return df


def build_acronym_map(df):
    """Create a mapping of each base name to its related acronyms."""
    mapping = defaultdict(set)
    for _, row in df.iterrows():
        acr = row.get("acronym")
        if pd.notna(acr) and str(acr).strip():
            mapping[row["base_name_norm"]].add(acr.strip())
    return mapping


def boost_acronym_similarity(df, sim, acr_map):
    """Increase similarity scores for names that share the same acronym."""
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i >= j:
                continue
            acr_i = row_i.get("acronym")
            acr_j = row_j.get("acronym")

            acrs_i = acr_map.get(row_i["base_name_norm"], set())
            acrs_j = acr_map.get(row_j["base_name_norm"], set())

            if (
                (acrs_i and acrs_j and acrs_i.intersection(acrs_j))
                or (acr_i in acrs_j)
                or (acr_j in acrs_i)
            ):
                sim[i, j] = sim[j, i] = max(sim[i, j], 0.95)
    return sim


def compute_hybrid_similarity(df, embedder):
    """Compute pairwise similarity using both character-based and embedding-based measures."""
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x_char = tfidf.fit_transform(df["base_name_norm"].tolist())
    sim_char = cosine_similarity(x_char)

    x_emb = embedder.encode(
        df["base_name_norm"].tolist(), convert_to_tensor=False, show_progress_bar=True
    )
    sim_sem = cosine_similarity(x_emb)

    sim = 0.5 * sim_char + 0.5 * sim_sem
    return sim


def cluster_names(df, embedder, sim_threshold=0.85):
    """
    Group similar dataset names into clusters based on hybrid similarity.

    This function takes a DataFrame of normalized dataset names and performs
    lightweight clustering using a similarity matrix derived from both
    character-level (TF-IDF) and semantic (embedding-based) representations.

    Steps:
        1. Clean and filter out noise (short or numeric-only names).
        2. Compute a hybrid similarity matrix using `compute_hybrid_similarity()`.
        3. Expand clusters using a simple breadth-first approach:
           - Each unvisited name starts a new cluster.
           - Any other name that exceeds the similarity threshold with it
             is added to the same cluster.
        4. Assign a unique cluster ID to each group of similar names.
    """
    # Step 1: Pre-filter invalid or noisy entries (too short, numeric-only)
    df = prefilter(df).reset_index(drop=True)

    # Step 2: Compute similarity matrix combining character and semantic similarity
    sim = compute_hybrid_similarity(df, embedder)

    # Step 3: Prepare containers for visited indices and cluster assignments
    visited = set()
    cluster_labels = np.full(len(df), -1)  # initialize all as unassigned
    cluster_id = 0

    # Step 4: Iterate through each name to form clusters
    for i in range(len(df)):
        # Skip already assigned items
        if i in visited:
            continue

        # Start a new cluster with the current item
        cluster_idx = [i]
        visited.add(i)

        # Compare current item with all others that are not yet visited
        for j in range(i + 1, len(df)):
            # If the similarity between i and j is above threshold, group them
            if j not in visited and sim[i, j] >= sim_threshold:
                cluster_idx.append(j)
                visited.add(j)

        # Assign the same cluster ID to all connected items
        cluster_labels[cluster_idx] = cluster_id
        cluster_id += 1

    # Step 5: Add the cluster labels to the DataFrame
    df["cluster"] = cluster_labels

    # Return the DataFrame with assigned clusters
    return df


def detect_country(raw: str, country_map: dict) -> str | None:
    """
    Detect the first matching country/demonym in text.
    Uses regex word boundaries to avoid substring false positives.
    Supports multi-word forms (e.g., 'viet nam').
    """
    clean = normalize(raw)

    for country, forms in country_map.items():
        for f in forms:
            norm_f = normalize(f)
            # \b handles word boundaries; \s+ allows internal spaces in multi-word forms
            pattern = r"\b" + re.sub(r"\s+", r"\\s+", re.escape(norm_f)) + r"\b"
            if re.search(pattern, clean):
                return country
    return None


def preprocess_cluster(df_cluster, country_map, country_pattern, lemmatizer, stopwords):
    """
    Preprocess cluster:
    - detect country from country_map
    - remove country/demonym
    - strip years/numbers/roman numerals
    - normalize text
    - remove stopwords + lemmatize
    """
    rows = []
    for _, row in df_cluster.iterrows():
        raw = row["raw_name"]
        count = row["count"]
        acronym = row.get("acronym", None)

        # detect country before stripping
        country = detect_country(raw, country_map)
        # base_name (country removed but before lemmatization)
        base = strip_country(raw, country_pattern)

        # normalized + lemmatized
        base_norm = base_name_norm(raw, country_pattern, lemmatizer, stopwords)

        rows.append(
            {
                "raw_name": raw,
                "count": int(count),
                "acronym": acronym if pd.notna(acronym) and str(acronym).strip() else None,
                "country": country,
                "base_name": base,
                "base_name_norm": base_norm,
            }
        )
    return pd.DataFrame(rows)


def group_prototypes(prototypes, sim_threshold=0.85):
    """
    Group similar dataset prototypes within each country based on name similarity.

    This function clusters prototype entries that share similar base names
    within the same country. It uses character-level TF-IDF similarity to
    identify likely variants or aliases of the same dataset family.
    For each cluster, the most frequent name (based on the 'count' field)
    is selected as the canonical "Prototype", and the rest are listed as "Aliases".
    """
    # Return empty if there are no prototypes to process
    if not prototypes:
        return []

    df = pd.DataFrame(prototypes)
    grouped = []

    # --- Group by country to avoid merging names from different contexts ---
    for country, subdf in df.groupby("country"):
        if country is None:
            continue  # Skip entries without a defined country

        names = subdf["base_name_norm"].tolist()
        if not names:
            continue  # Skip empty country groups

        # --- Step 1: Compute character-level TF-IDF similarity ---
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        x = vec.fit_transform(names)
        sim = cosine_similarity(x)

        visited = set()

        # --- Step 2: Simple greedy clustering by similarity threshold ---
        for i, name in enumerate(names):
            if i in visited:
                continue

            cluster_idx = [i]
            visited.add(i)

            # Compare each unvisited name and group if similarity is high
            for j in range(i + 1, len(names)):
                if j not in visited and sim[i, j] >= sim_threshold:
                    cluster_idx.append(j)
                    visited.add(j)

            # --- Step 3: Pick the canonical (most frequent) record ---
            sub = subdf.iloc[cluster_idx]
            canonical_row = sub.loc[sub["count"].idxmax()]
            canonical = canonical_row.to_dict()

            # Collect other variants (aliases) within the same cluster
            aliases = [
                row.to_dict()
                for _, row in sub.iterrows()
                if row["raw_name"] != canonical["raw_name"]
            ]

            # --- Step 4: Save cluster result ---
            grouped.append({"Prototype": canonical, "Aliases": aliases})

    # Return list of grouped prototypes with their alias sets
    return grouped


def has_year(name: str) -> bool:
    """Detect 4-digit years or Roman numerals in a string."""
    return bool(re.search(r"\b(19|20)\d{2}\b", name)) or bool(
        re.search(r"\b[ivxlcdm]{1,4}\b", name, flags=re.I)
    )


def pick_canonical(subdf):
    """
    Pick a canonical entry from a cluster:
    1. Prefer entries with no country AND no year.
    2. If none, prefer entries with no country (even if year exists).
    3. If none, fallback to overall highest count.

    Within each group:
    - Prefer entries with acronym.
    - Then highest count.
    """
    subdf = subdf.copy()
    subdf["has_year"] = subdf["raw_name"].apply(has_year)
    subdf["has_country"] = subdf["country"].notna() & (subdf["country"] != "")

    # Rule 1: no country, no year
    no_country_no_year = subdf[(~subdf["has_country"]) & (~subdf["has_year"])]
    if not no_country_no_year.empty:
        with_acr = no_country_no_year[
            no_country_no_year["acronym"].notna()
            & (no_country_no_year["acronym"].str.strip() != "")
        ]
        if not with_acr.empty:
            return with_acr.loc[with_acr["count"].idxmax()]
        return no_country_no_year.loc[no_country_no_year["count"].idxmax()]

    # Rule 2: no country (year may exist)
    no_country = subdf[~subdf["has_country"]]
    if not no_country.empty:
        with_acr = no_country[
            no_country["acronym"].notna() & (no_country["acronym"].str.strip() != "")
        ]
        if not with_acr.empty:
            return with_acr.loc[with_acr["count"].idxmax()]
        return no_country.loc[no_country["count"].idxmax()]

    # Rule 3: fallback to overall
    with_acr = subdf[subdf["acronym"].notna() & (subdf["acronym"].str.strip() != "")]
    if not with_acr.empty:
        return with_acr.loc[with_acr["count"].idxmax()]
    return subdf.loc[subdf["count"].idxmax()]


def camelcase_canonical(name: str) -> str:
    """
    Convert a canonical name into CamelCase / TitleCase style.
    Keeps acronyms uppercase and common stopwords lowercase (unless first word).
    """
    stopwords = {"of", "and", "in", "for", "on", "at", "with", "the"}
    words = name.strip().split()
    fixed = []
    for i, w in enumerate(words):
        lw = w.lower()
        if w.isupper():  # keep acronyms like CPI, GHS
            fixed.append(w)
        elif len(w) == 1:  # single letters like U.
            fixed.append(w.upper())
        elif lw in stopwords and i != 0:  # stopwords lowercase unless first word
            fixed.append(lw)
        else:
            fixed.append(w.capitalize())
    return " ".join(fixed)


def build_families(df, sim_threshold=0.85):
    """
    Build dataset "families" by clustering similar names and organizing them
    into canonical forms, aliases, and country-specific prototypes.

    This function identifies groups of closely related dataset names using
    character-level TF-IDF similarity. Within each group, it selects a
    representative (canonical) name and assigns the rest as either aliases
    (similar names without country tags) or prototypes (country-specific
    variants). Each resulting "family" provides a structured view of related
    dataset references across countries and naming styles.
    """
    # --- Step 1: Compute pairwise similarity among normalized names ---
    names = df["base_name_norm"].tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x = vec.fit_transform(names)
    sim = cosine_similarity(x)

    visited = set()
    families = []

    # --- Step 2: Iterate through all names to form similarity-based clusters ---
    for i, name in enumerate(names):
        if i in visited:
            continue

        # Initialize a new cluster starting from name i
        cluster_idx = [i]
        visited.add(i)

        # Add all names that are similar enough (above threshold)
        for j in range(i + 1, len(names)):
            if j not in visited and sim[i, j] >= sim_threshold:
                cluster_idx.append(j)
                visited.add(j)

        subdf = df.iloc[cluster_idx]

        # --- Step 3: Determine the canonical name for this cluster ---
        # pick_canonical() selects the most representative record
        canonical_row = pick_canonical(subdf)
        canonical = canonical_row.to_dict()

        # Ensure consistent formatting for the canonical display name
        canonical["raw_name"] = camelcase_canonical(canonical["raw_name"])

        aliases, prototypes = [], []

        # --- Step 4: Classify remaining members as aliases or prototypes ---
        for _, row in subdf.iterrows():
            if row["raw_name"] == canonical["raw_name"]:
                continue  # skip the canonical itself

            # Compute direct similarity to the canonical for filtering
            score = cosine_similarity(
                x[[df.index.get_loc(canonical_row.name)]], x[[df.index.get_loc(row.name)]]
            )[0, 0]

            entry = row.to_dict()

            # Country-specific entries become prototypes
            if score >= sim_threshold and not row["country"]:
                aliases.append(entry)
            elif row["country"]:
                prototypes.append(entry)
            else:
                aliases.append(entry)

        # --- Step 5: Group prototypes further by country similarity ---
        families.append(
            {
                "Canonical": canonical,
                "Aliases": aliases,
                "Prototypes": group_prototypes(prototypes, sim_threshold),
            }
        )

    # --- Step 6: Return full list of dataset families ---
    return families


def format_hierarchy(family: dict) -> str:
    lines = []
    lines.append(f"{format_name(family['Canonical'])}  <-- Canonical")

    if family.get("Aliases"):
        alias_str = ", ".join(f'"{format_name(a)}"' for a in family["Aliases"])
        lines.append(f"├─ Aliases: {alias_str}")

    if family.get("Prototypes"):
        lines.append("├─ Prototypes:")
        for proto in family["Prototypes"]:
            pname = format_name(proto["Prototype"])
            lines.append(f"│   ├─ {pname}")
            if proto["Aliases"]:
                alias_str = ", ".join(f'"{format_name(a)}"' for a in proto["Aliases"])
                lines.append(f"│   │    ├─ Aliases: {alias_str}")

    return "\n".join(lines)


def is_acronym_variant(word: str, acr: str) -> bool:
    """
    Decide if `word` is a legitimate variant of acronym `acr`.
    """
    word = word.lower()
    acr = acr.lower()

    if word == acr:
        return True
    if word.startswith(acr):  # LSMSs, DHS2010
        return True
    if re.search(rf"\b{acr}\b", word):  # mini-DHS, DHS survey
        return True
    return False


def learn_family_keys(families, sim_threshold=85, sem_threshold=0.8):
    """
    Learn a mapping from dataset name variants to their canonical family names.

    This function builds a dictionary that links all known variants—aliases,
    prototypes, and acronym forms—to their corresponding canonical dataset name.
    It ensures that references to the same dataset across different spellings,
    countries, or formats are unified under a single representative name.

    Steps
    -----
    1. Loop through each family entry produced by `build_families()`.
    2. Extract the canonical dataset name, normalized base name, and any acronym.
    3. Collect all related variants from:
        - Aliases (same dataset, different phrasing)
        - Prototypes (country-specific versions, including their aliases)
    4. Compare each variant to the canonical form using:
        - Acronym equivalence (`is_acronym_variant()`), or
        - Fuzzy string similarity (via `fuzz.ratio`)
    5. Add all qualified variants to a shared dictionary (`family_keys`)
       mapping variant → canonical name.

    Parameters
    ----------
    families : list of dict
        Output from `build_families()`. Each element must include:
        - "Canonical": dict with 'raw_name', optional 'acronym', and 'base_name_norm'.
        - "Aliases": list of similar names.
        - "Prototypes": list of country-specific subgroups.
    sim_threshold : int, optional (default=85)
        Minimum fuzzy string similarity (0–100) between a variant and the
        canonical form for them to be treated as equivalent.
    sem_threshold : float, optional (default=0.8)
        Reserved for potential semantic similarity checks (not currently used).

    Returns
    -------
    dict
        A dictionary mapping normalized or variant names to their unified
        canonical names. Example:
        {
            "demographic_and_health_survey": "Demographic and Health Survey (DHS)",
            "dhs": "Demographic and Health Survey (DHS)",
            "demographic_health_survey_kenya": "Demographic and Health Survey (DHS)"
        }

    Notes
    -----
    - Helps deduplicate and standardize dataset references across sources.
    - The resulting `family_keys` dictionary can be used to merge records,
      build canonical hierarchies, or evaluate mention-level consistency.
    - Uses fuzzy string matching to tolerate minor textual differences
      (e.g., spacing, punctuation, or pluralization).
    """

    family_keys = {}

    # --- Step 1: Iterate over each dataset family ---
    for fam in families:
        cname = fam["Canonical"]["raw_name"]
        base_norm = fam["Canonical"].get("base_name_norm", cname.lower())
        acr = fam["Canonical"].get("acronym")

        # --- Step 2: Construct the canonical display name (with acronym if available) ---
        canonical_name = cname
        if acr and str(acr).strip():
            canonical_name = f"{cname} ({acr})"

        # --- Step 3: Collect all variant names from aliases and prototypes ---
        variants = []
        counts = Counter()

        # Add aliases (direct variants)
        for alias in fam.get("Aliases", []):
            norm = alias.get("base_name_norm", alias["raw_name"].lower())
            variants.append(norm)
            counts[norm] += alias.get("count", 1)

        # Add prototypes and their aliases
        for proto in fam.get("Prototypes", []):
            pnorm = proto["Prototype"].get("base_name_norm", proto["Prototype"]["raw_name"].lower())
            variants.append(pnorm)
            counts[pnorm] += proto["Prototype"].get("count", 1)

            for a in proto.get("Aliases", []):
                anorm = a.get("base_name_norm", a["raw_name"].lower())
                variants.append(anorm)
                counts[anorm] += a.get("count", 1)

        # --- Step 4: Map the canonical base name to its full canonical form ---
        family_keys[base_norm] = canonical_name

        # --- Step 5: Map each variant if it's similar enough to the canonical ---
        for v in set(variants):
            # Check if the variant is a known acronym form
            if acr and is_acronym_variant(v, acr):
                family_keys[v] = canonical_name
            else:
                # Apply fuzzy matching for close string similarity
                match = process.extractOne(v, [base_norm], scorer=fuzz.ratio)
                if match and match[1] >= sim_threshold:
                    family_keys[v] = canonical_name

        # --- Step 6: Map the acronym itself for quick lookup ---
        if acr and str(acr).strip():
            family_keys[acr.lower()] = canonical_name

    # --- Step 7: Return the full variant-to-canonical mapping ---
    return family_keys


def merge_acronyms(families, sim_threshold=0.8):
    """
    Merge dataset families whose canonical names are connected by shared acronyms.

    This function looks for families that define both a long-form dataset name
    (e.g., “Demographic and Health Survey”) and an acronym version (e.g., “DHS”).
    When the acronym appears within another family’s canonical name, the two
    families are merged into a single entry. The result combines all aliases and
    prototypes under one unified family.

    Steps
    -----
    1. Loop through each family and check if its canonical record defines an acronym.
    2. For each acronym-defined family:
        - Find other families whose canonical names contain that acronym.
        - Merge their canonical record, aliases, and prototypes into the current one.
    3. Families without acronyms are kept as-is.

    Parameters
    ----------
    families : list of dict
        A list of dataset families, as produced by `build_families()`.
        Each family must contain:
        - "Canonical": dict with 'raw_name' and optional 'acronym'.
        - "Aliases": list of related name variants.
        - "Prototypes": list of country-specific subgroups.
    sim_threshold : float, optional (default=0.8)
        Reserved for possible future similarity-based matching.
        Currently unused—acronym matching is based on substring checks.

    Returns
    -------
    list of dict
        A list of merged families where acronym-linked entries are unified.
        Each dictionary retains the structure:
        {
            "Canonical": {...},
            "Aliases": [...],
            "Prototypes": [...]
        }

    Notes
    -----
    - Merging is shallow: when two families share an acronym, the “long-form”
      (the one that defines the acronym) absorbs the other.
    - The function avoids merging a family more than once by tracking used indices.
    - Designed for post-processing after canonical families have already been built.
    """

    merged = []
    used = set()

    # --- Step 1: Iterate through each family and look for acronym-defined entries ---
    for i, fam in enumerate(families):
        if i in used:
            continue  # skip if this family has already been merged

        canonical = fam["Canonical"]
        acr = canonical.get("acronym")

        # --- Step 2: If this canonical defines an acronym, search for matching families ---
        if acr and str(acr).strip():
            longform_family = fam

            for j, other in enumerate(families):
                if j == i or j in used:
                    continue

                other_name = other["Canonical"]["raw_name"]
                other_base = other["Canonical"].get("base_name_norm", other_name.lower())

                # --- Step 3: Check if the acronym appears in the other canonical’s name ---
                if acr.lower() in other_name.lower() or acr.lower() in other_base:
                    # Merge the other family into the current one
                    longform_family["Aliases"].append(other["Canonical"])
                    longform_family["Aliases"].extend(other.get("Aliases", []))
                    longform_family["Prototypes"].extend(other.get("Prototypes", []))
                    used.add(j)

            # Save the merged long-form family
            merged.append(longform_family)
            used.add(i)

        else:
            # --- Step 4: Families without acronyms are kept unchanged ---
            merged.append(fam)
            used.add(i)

    # --- Step 5: Return the final merged list of families ---
    return merged


def consolidate_families(families, family_keys, sim_threshold=85):
    """
    Consolidate fragmented dataset families into unified canonical groups.

    This function takes a list of partially overlapping families and merges
    them based on a learned mapping (`family_keys`) that links variant names
    to their canonical dataset forms. It ensures that any families referring
    to the same dataset—under slightly different spellings or acronyms—are
    collapsed into a single, unified structure.

    Steps
    -----
    1. Loop through each family and extract its canonical base name.
    2. Check if that base name appears (exactly or fuzzily) in `family_keys`.
       - If found, assign it to the mapped canonical name.
       - If not found, use its own canonical name as a fallback.
    3. Merge all associated aliases and prototypes under the unified canonical.
    4. Return a list of consolidated families, each representing a single
       canonical dataset and its combined variants.

    Parameters
    ----------
    families : list of dict
        A list of dataset families (from `build_families()` or `merge_acronyms()`),
        where each family includes:
        - "Canonical": dict containing 'raw_name' and 'base_name_norm'.
        - "Aliases": list of related name variants.
        - "Prototypes": list of country-specific groups.
    family_keys : dict
        A mapping of normalized or variant names to their canonical dataset
        names, as produced by `learn_family_keys()`.
    sim_threshold : int, optional (default=85)
        Minimum fuzzy string similarity (0–100) used when matching a family's
        base name to a known key in `family_keys`.

    Returns
    -------
    list of dict
        A list of consolidated canonical families, where each element contains:
        - "Canonical": dict with the unified canonical name.
        - "Aliases": combined list of aliases from all merged families.
        - "Prototypes": combined list of prototypes from all merged families.

    Notes
    -----
    - This step is the final layer of hierarchy consolidation, used after
      individual family structures have been formed and acronym-linked
      entries merged.
    - The resulting list is clean, deduplicated, and ready for export,
      evaluation, or graph construction.
    """

    # --- Step 1: Initialize storage for merged canonical families ---
    consolidated = defaultdict(lambda: {"Canonical": None, "Aliases": [], "Prototypes": []})

    # --- Step 2: Process each input family ---
    for fam in families:
        cname = fam["Canonical"]["raw_name"]
        base_norm = fam["Canonical"].get("base_name_norm", cname.lower())

        # --- Step 3: Try exact or fuzzy matching against FAMILY_KEYS ---
        if base_norm in family_keys:
            target = family_keys[base_norm]
        else:
            # Use fuzzy string matching when exact match fails
            match = process.extractOne(base_norm, list(family_keys.keys()), scorer=fuzz.ratio)
            if match and match[1] >= sim_threshold:
                target = family_keys[match[0]]
            else:
                target = cname  # fallback if no close match found

        # --- Step 4: Initialize canonical entry if not yet created ---
        if not consolidated[target]["Canonical"]:
            consolidated[target]["Canonical"] = {"name": target}

        # --- Step 5: Merge all aliases and prototypes under this canonical ---
        consolidated[target]["Aliases"].extend(fam.get("Aliases", []))
        consolidated[target]["Prototypes"].extend(fam.get("Prototypes", []))

    # --- Step 6: Return the final list of unified family entries ---
    return list(consolidated.values())


def format_name(entry):
    """Format name with acronym if present (skip NaN/None)."""
    if not entry:
        return ""
    # Try raw_name first, else fall back to name
    raw = entry.get("raw_name") or entry.get("name") or ""
    acr = entry.get("acronym")
    if acr and str(acr).strip():
        return f"{raw} ({acr})"
    return raw


## setup


def initialize_environment(
    country_map_path=None,
    min_population=500_000,
    embedder_model="avsolatorio/GIST-all-MiniLM-L6-v2",
):
    """
    Initialize shared resources and environment settings for the harmonization pipeline.

    Loads and prepares all reusable objects such as country maps, regex patterns,
    stopword lists, lemmatizer, and sentence embedding models.

    Parameters
    ----------
    country_map_path : str
        Path to the JSON file containing country and city mappings.
    min_population : int, optional (default=500_000)
        Minimum population threshold for filtering cities in the country map.
    embedder_model : str, optional
        Name or path of the SentenceTransformer model to load for semantic similarity.

    Returns
    -------
    dict
        Dictionary containing all initialized resources:
        {
            "country_map": dict,
            "country_pattern": regex,
            "lemmatizer": WordNetLemmatizer,
            "STOPWORDS": set,
            "embedder": SentenceTransformer
        }
    """
    # --- Resolve default path ---
    if country_map_path is None:
        # Get the parent directory of extractors (ai4data package root)
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        country_map_path = os.path.join(package_root, "assets", "country_map.json")

    # --- Load and preprocess country map ---
    with open(country_map_path, "r", encoding="utf-8") as f:
        country_map = json.load(f)
    country_map = build_country_map_with_cities_only(country_map, min_population=min_population)

    # --- Build compiled regex pattern for country matching ---
    country_pattern = build_country_regex(country_map)

    # --- NLTK setup ---
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words("english"))
    domain_stopwords = {
        "data",
        "dataset",
        "datasets",
        "survey",
        "surveys",
        "study",
        "studies",
        "statistics",
        "database",
        "project",
        "program",
        "report",
        "data sets",
        "data set",
    }
    stopwords_set.update(domain_stopwords)
    # --- Load embedding model ---
    embedder = SentenceTransformer(embedder_model)

    print("Environment initialized successfully.")
    print(f"• Loaded {len(country_map)} countries")
    print(f"• Using model: {embedder_model}")
    print(f"• Stopwords loaded: {len(stopwords_set)}")

    return {
        "country_map": country_map,
        "country_pattern": country_pattern,
        "lemmatizer": lemmatizer,
        "stopwords": stopwords_set,
        "embedder": embedder,
    }


## PIPELINE RUN
def load_and_preprocess(all_fnames, country_map_path, lemmatizer, stopwords_set, named_only=True):
    """Load and preprocess S2ORC format data."""
    from tqdm.auto import tqdm

    df = pd.DataFrame()
    for fn in tqdm(all_fnames, total=len(all_fnames), desc="Loading initial data"):
        # for s2orc
        obj = load_json_data(fn)
        res_now = obj["data"] if "data" in obj else obj
        res_now = [r for r in res_now if r["text"] != ""]
        df = pd.concat([df, pd.DataFrame(res_now)], axis=0)
    if named_only:
        df = df.loc[df["label"] == "named"]
    df = df.rename(columns={"text": "datasets"})
    df = df.reset_index(drop=True)
    df["datasets"] = df["datasets"].astype(str).str.strip()
    df = df.loc[df.groupby("datasets")["count"].idxmax()].reset_index(drop=True)
    df = df.sort_values("count", ascending=False)

    df = df[["datasets", "acronym", "count"]].rename(columns={"acronym": "acronyms"})

    df["acronym"] = df.apply(
        lambda row: pick_plausible_acronym(row.get("datasets", ""), row.get("acronyms", None)),
        axis=1,
    )
    with open(country_map_path, "r", encoding="utf-8") as f:
        country_map = json.load(f)

    country_map = build_country_map_with_cities_only(country_map, min_population=500000)
    country_pattern = build_country_regex(country_map)

    df["base_name_norm"] = df["datasets"].apply(
        lambda x: base_name_norm(x, country_pattern, lemmatizer, stopwords_set)
    )

    return df, country_pattern


def load_dedup_and_preprocess(
    dedup_folders, country_map_path, lemmatizer, stopwords_set, dataset_tags=None
):
    """
    Load and preprocess dedup format data.

    Parameters
    ----------
    dedup_folders : list of str
        List of paths to dedup folders containing JSON files.
    country_map_path : str
        Path to country_map.json.
    lemmatizer : WordNetLemmatizer
        NLTK lemmatizer instance.
    STOPWORDS : set
        Set of stopwords to remove.
    dataset_tags : list of str, optional
        List of dataset tags to include (e.g., ['named'], ['named', 'descriptive']).
        If None, defaults to ['named'] only. 'non-dataset' tags are always excluded.

    Returns
    -------
    tuple
        (DataFrame, country_pattern)
    """
    from tqdm.auto import tqdm

    # Load all dedup files from the folders
    all_dedup_files = []
    for folder in tqdm(dedup_folders, desc="Loading dedup folders"):
        files = glob.glob(os.path.join(folder, "*.json"))
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_dedup_files.append(json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    # Extract dataset mentions
    df = extract_dataset_mentions(all_dedup_files)

    if df.empty:
        print("Warning: No dataset mentions found in dedup files")
        return pd.DataFrame(columns=["datasets", "acronym", "count", "base_name_norm"]), None

    # Set default tags if not provided
    if dataset_tags is None:
        dataset_tags = ["named"]

    # Always exclude 'non-dataset' tags
    df = df[df["label"] != "non-dataset"]

    # Filter by specified dataset tags
    if dataset_tags:
        df = df[df["label"].isin(dataset_tags)]

    # Rename for consistency with S2ORC format
    # df already has 'datasets' column from extract_dataset_mentions
    df = df.reset_index(drop=True)
    df["datasets"] = df["datasets"].astype(str).str.strip()
    df = df.loc[df.groupby("datasets")["count"].idxmax()].reset_index(drop=True)
    df = df.sort_values("count", ascending=False)

    # Prepare acronyms column
    df = df[["datasets", "acronym", "count"]].rename(columns={"acronym": "acronyms"})

    df["acronym"] = df.apply(
        lambda row: pick_plausible_acronym(row.get("datasets", ""), row.get("acronyms", None)),
        axis=1,
    )

    # Load country map
    with open(country_map_path, "r", encoding="utf-8") as f:
        country_map = json.load(f)

    country_map = build_country_map_with_cities_only(country_map, min_population=500000)
    country_pattern = build_country_regex(country_map)

    df["base_name_norm"] = df["datasets"].apply(
        lambda x: base_name_norm(x, country_pattern, lemmatizer, stopwords_set)
    )

    return df, country_pattern


def create_smart_batches(cluster_sizes, target_batch_size=300, small_threshold=10):
    batches, current_batch, current_size = [], [], 0
    small_clusters, large_clusters = [], []

    for _, row in cluster_sizes.iterrows():
        cid, size = row["cluster"], row["n_entries"]
        if size <= small_threshold:
            small_clusters.append(cid)
        else:
            large_clusters.append((cid, size))

    # Combine small clusters into balanced batches
    for cid in small_clusters:
        current_batch.append(cid)
        current_size += cluster_sizes.loc[cluster_sizes["cluster"] == cid, "n_entries"].values[0]
        if current_size >= target_batch_size:
            batches.append(current_batch)
            current_batch, current_size = [], 0
    if current_batch:
        batches.append(current_batch)

    # Handle large clusters separately
    for cid, size in large_clusters:
        if size < target_batch_size // 2 and batches and len(batches[-1]) == 1:
            batches[-1].append(cid)
        else:
            batches.append([cid])

    print(f"Total batches formed: {len(batches)}")
    return batches


def integrate_families(
    families_now, family_keys_now, all_families, all_family_keys, sim_threshold=85
):
    """
    Merge incremental families into the master families with fuzzy canonical matching. This is using the base_name_norm #raf.
    """
    for fam in families_now:
        cname = fam["Canonical"].get("raw_name") or fam["Canonical"].get("name")
        base_norm = fam["Canonical"].get("base_name_norm", cname.lower() if cname else "")
        if not cname:
            continue

        match = process.extractOne(base_norm, list(all_family_keys.keys()), scorer=fuzz.ratio)
        if match and match[1] >= sim_threshold:
            target_canonical = all_family_keys[match[0]]
            for existing_fam in all_families:
                existing_canonical = existing_fam["Canonical"].get("raw_name") or existing_fam[
                    "Canonical"
                ].get("name")
                if existing_canonical == target_canonical:
                    existing_fam["Aliases"].extend(fam.get("Aliases", []))
                    existing_fam["Prototypes"].extend(fam.get("Prototypes", []))
                    break
        else:
            all_families.append(fam)

        all_family_keys.update(family_keys_now)

    return all_families, all_family_keys


def run_incremental_pipeline(
    initial_folder=None,
    output_dir=None,
    country_map_path=None,
    env=None,
    initial_wave=50,
    incremental_wave=10,
    sim_threshold=0.5,
    data_format="s2orc",
    dataset_tags=None,
):
    """
    Run the incremental dataset-family harmonization pipeline.

    This function processes batches of clustered dataset mentions in waves
    (initial and incremental), building and merging hierarchical families
    of related dataset names.

    It requires an `initial_folder` structured such that each subfolder
    contains a `dedup/` directory with JSON files for dataset mentions.

    Example expected structure
    --------------------------
    main_folder/
    ├── subfolder_001/
    │   └── dedup/
    │       ├── file1.json
    │       ├── file2.json
    │       └── ...
    ├── subfolder_002/
    │   └── dedup/
    │       ├── file1.json
    │       └── ...

    Parameters
    ----------
    initial_folder : str
        **Required.** Path to the root directory containing subfolders with
        `dedup/` JSON files (e.g. `"./regrouped_s2orc/"`).
        Each `dedup/` folder must contain pre-processed JSON files.
    output_dir : str, optional
        Directory where output JSON files will be saved.
        Defaults to `./harmonization_outputs/` if not provided.
    country_map_path : str, optional
        Path to `country_map.json`. Defaults to `./assets/country_map.json`.
    env : dict, optional
        Pre-initialized environment dictionary from `initialize_environment()`.
        If None, the environment will be created automatically.
    initial_wave : int, optional (default=30)
        Number of folders to process in the initial wave.
    incremental_wave : int, optional (default=10)
        Number of folders to process per incremental wave.
    sim_threshold : float, optional (default=0.5)
        Similarity threshold used for name clustering.
    data_format : str, optional (default="s2orc")
        Data format: "s2orc" for S2ORC format or "dedup" for dedup output format.
    dataset_tags : list of str, optional
        List of dataset tags to include (e.g., ['named'], ['named', 'descriptive']).
        If None, defaults to ['named'] only. 'non-dataset' tags are always excluded.
        Only applies when data_format="dedup".

    Raises
    ------
    ValueError
        If `initial_folder` is not provided or does not exist.

    Returns
    -------
    None
        Writes multiple output JSON files under `output_dir/`:
        - `all_families_initial.json`
        - `all_family_keys_initial.json`
        - Incremental wave outputs (e.g. `all_families_wave1.json`)
        - Final integrated results (`all_families_master.json`)
    """

    # --- Validate required input folder ---
    if initial_folder is None:
        raise ValueError(
            " `initial_folder` is required. It must point to a directory that "
            "contains subfolders with `dedup/*.json` files.\n"
            "Example: regrouped_s2orc/subfolder_001/dedup/file1.json"
        )

    if not os.path.exists(initial_folder):
        raise ValueError(f"The provided initial_folder does not exist: {initial_folder}")

    # --- Resolve optional defaults ---
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "harmonization_outputs")
    if country_map_path is None:
        # Get the parent directory of extractors (ai4data package root)
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        country_map_path = os.path.join(package_root, "assets", "country_map.json")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input folder: {initial_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Country map: {country_map_path}")

    # --- Initialize environment if not provided ---
    if env is None:
        print("Initializing environment...")
        env = initialize_environment(country_map_path)
    else:
        print("Using provided environment.")

    country_map = env["country_map"]
    country_pattern = env["country_pattern"]
    lemmatizer = env["lemmatizer"]
    stopwords_set = env["stopwords"]
    embedder = env["embedder"]

    # Detect folders based on data format
    if data_format == "dedup":
        # For dedup format, look for dedup folders directly
        all_folders = get_dedup_folder_structure(initial_folder)
    else:
        # For S2ORC format, use the original pattern
        all_folders = sorted(glob.glob(f"{initial_folder}*/dedup/"))

    total_folders = len(all_folders)
    print(f"Total folders detected: {total_folders}")
    print(f"Data format: {data_format}")

    initial_folders = all_folders[:initial_wave]
    remaining_folders = all_folders[initial_wave:]

    # --- INITIAL WAVE ---
    print("\n Running initial wave...")

    # Load data based on format
    if data_format == "dedup":
        df, country_pattern = load_dedup_and_preprocess(
            initial_folders, country_map_path, lemmatizer, stopwords_set, dataset_tags=dataset_tags
        )
    else:
        all_fnames = []
        for folder in initial_folders:
            all_fnames.extend(glob.glob(os.path.join(folder, "*.json")))
        df, country_pattern = load_and_preprocess(
            all_fnames, country_map_path, lemmatizer, stopwords_set
        )

    df_clustered = cluster_names(
        df[["datasets", "count", "acronym", "base_name_norm"]], embedder, sim_threshold
    )

    cluster_sizes = (
        df_clustered.groupby("cluster")
        .size()
        .reset_index(name="n_entries")
        .sort_values("n_entries", ascending=False)
    )
    batches = create_smart_batches(cluster_sizes)

    all_families, all_family_keys = [], {}
    for batch_idx, batch in enumerate(tqdm(batches, desc="Initial hierarchy batches")):
        df_batch = df_clustered[df_clustered["cluster"].isin(batch)].rename(
            columns={"datasets": "raw_name"}
        )
        if len(df_batch) < 2:
            continue

        df_pre = preprocess_cluster(
            df_batch, country_map, country_pattern, lemmatizer, stopwords_set
        )
        families = build_families(df_pre, sim_threshold=0.85)
        families = merge_acronyms(families)
        family_keys = learn_family_keys(families)
        families = consolidate_families(families, family_keys)

        all_families.extend(families)
        all_family_keys.update(family_keys)

    save_json(all_families, os.path.join(output_dir, "all_families_initial.json"))
    save_json(all_family_keys, os.path.join(output_dir, "all_family_keys_initial.json"))
    print(f"\nInitial wave completed: {len(all_families)} families")

    # --- INCREMENTAL WAVES ---
    for wave_idx in range(0, len(remaining_folders), incremental_wave):
        wave_folders = remaining_folders[wave_idx : wave_idx + incremental_wave]
        wave_num = wave_idx // incremental_wave + 1
        print(f"\nRunning incremental wave {wave_num} ({len(wave_folders)} folders)")

        # Load data based on format
        if data_format == "dedup":
            new_df, _ = load_dedup_and_preprocess(
                wave_folders, country_map_path, lemmatizer, stopwords_set, dataset_tags=dataset_tags
            )
        else:
            all_fnames_new = []
            for folder in wave_folders:
                all_fnames_new.extend(glob.glob(os.path.join(folder, "*.json")))
            new_df, _ = load_and_preprocess(
                all_fnames_new, country_map_path, lemmatizer, stopwords_set
            )

        df_clustered_new = cluster_names(
            new_df[["datasets", "count", "acronym", "base_name_norm"]], embedder, sim_threshold
        )

        cluster_sizes = (
            df_clustered_new.groupby("cluster")
            .size()
            .reset_index(name="n_entries")
            .sort_values("n_entries", ascending=False)
        )
        batches = create_smart_batches(cluster_sizes)

        for batch_idx, cluster_group in enumerate(tqdm(batches, desc=f"Wave {wave_num} batches")):
            subdf = df_clustered_new[df_clustered_new["cluster"].isin(cluster_group)]
            subdf = subdf.rename(columns={"datasets": "raw_name"})
            if subdf.empty:
                continue

            subdf = preprocess_cluster(
                subdf, country_map, country_pattern, lemmatizer, stopwords_set
            )
            families_now = build_families(subdf, sim_threshold=0.85)
            families_now = merge_acronyms(families_now)
            family_keys_now = learn_family_keys(families_now)

            all_families, all_family_keys = integrate_families(
                families_now, family_keys_now, all_families, all_family_keys, sim_threshold=85
            )

        save_json(all_families, os.path.join(output_dir, f"all_families_wave{wave_num}.json"))
        save_json(all_family_keys, os.path.join(output_dir, f"all_family_keys_wave{wave_num}.json"))
        print(f"Completed wave {wave_num}: {len(all_families)} total families")

    print("\nFinal integration complete!")
    save_json(all_families, os.path.join(output_dir, "all_families_master.json"))
    save_json(all_family_keys, os.path.join(output_dir, "all_family_keys_master.json"))
