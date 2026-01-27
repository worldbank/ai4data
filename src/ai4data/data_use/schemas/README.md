# Dataset Extraction Schema

This document provides comprehensive documentation for the **Dataset Extraction Schema** used in the `ai4data` library.

## Overview

The extraction schema is designed to identify and extract structured information about dataset mentions from academic papers, reports, and other documents. It uses a GLiNER2-based model to perform named entity recognition with a structured schema.

## Schema Structure

The schema is defined in `dataset_schema.py` and extracts the following information for each dataset mention:

### Core Dataset Identity

#### 1. `dataset_name` (string)
- **Description**: The extracted name of the dataset as mentioned in the text
- **Type**: String
- **Default Threshold**: 0.85 (customizable)
- **Examples**:
  - Formal titles: "Demographic and Health Survey"
  - Informal references: "household survey data"
- **Notes**: May vary depending on the tagging classification

#### 2. `tag` (categorical)
- **Description**: Classification of the dataset mention type
- **Type**: String (categorical)
- **Choices**:
  - `"named"`: Formal dataset titles (e.g., "World Bank Living Standards Measurement Study")
  - `"descriptive"`: Unnamed but clearly defined datasets (e.g., "administrative employment records")
  - `"vague"`: Ambiguous references to data sources (e.g., "survey data")
  - `"non-dataset"`: When the term does not function as a dataset in context or is empty
- **Use Case**: Helps filter and prioritize dataset mentions based on specificity

---

### Data Characteristics

#### 3. `data_type` (string)
- **Description**: High-level category of the dataset based on its nature
- **Type**: String
- **Common Values**:
  - `"survey"`: Structured questionnaires
  - `"report"`: Compiled statistical summaries
  - `"program"`: Operational or monitoring systems
  - `"census"`: Population-wide enumerations
  - `"system"`: Administrative or information systems
- **Examples**:
  - "Demographic and Health Survey" → `"survey"`
  - "National Census 2020" → `"census"`
  - "Health Information Management System" → `"system"`

#### 4. `description` (string)
- **Description**: Short description of the type of data contained in the dataset
- **Type**: String
- **Examples**:
  - "household data"
  - "crime reports"
  - "satellite imagery"
  - "employment indicators"
  - "administrative microdata"
- **Notes**: Describes the data **content**, not the dataset category

---

### Provenance Metadata

#### 5. `acronym` (string)
- **Description**: The acronym associated with the dataset, if explicitly mentioned
- **Type**: String
- **Default Threshold**: 0.85 (customizable)
- **Examples**:
  - "High-Frequency Survey" → `"HFS"`
  - "Demographic and Health Survey" → `"DHS"`
  - "Living Standards Measurement Study" → `"LSMS"`

#### 6. `producer` (string)
- **Description**: The institution or organization that produced, collected, or published the dataset
- **Type**: String
- **Examples**:
  - "National Statistics Office"
  - "World Bank"
  - "Ministry of Health"
  - "Philippine Statistics Authority"
  - "International Labour Organization"

#### 7. `publication_year` (string)
- **Description**: The year the dataset was released or published
- **Type**: String
- **Notes**: Distinct from `reference_year`, which refers to when data were collected
- **Example**: A 2018 survey published in 2019 would have `publication_year = "2019"`

#### 8. `reference_year` (string)
- **Description**: The year or time period the data refer to
- **Type**: String
- **Notes**: The year of data collection (e.g., 2018 survey year, 2020 census year)
- **Example**: A 2018 survey published in 2019 would have `reference_year = "2018"`

#### 9. `reference_population` (string)
- **Description**: The target population covered by the dataset
- **Type**: String
- **Examples**:
  - "households"
  - "migrant workers"
  - "urban residents"
  - "women aged 15–49"
  - "children under 5 years"

---

### Usage Classification

#### 10. `is_used` (categorical)
- **Description**: Indicates whether the dataset is actually used in empirical analysis
- **Type**: String (categorical)
- **Choices**:
  - `"true"`: Dataset is used in the empirical analysis
  - `"false"`: Only mentioned for context, comparison, or narrative framing
- **Use Case**: Distinguishes between datasets that are actively analyzed vs. merely cited

#### 11. `mention_context` (categorical)
- **Description**: Describes how the dataset is used in the document
- **Type**: String (categorical)
- **Choices**:
  - `"primary"`: Main analytical dataset
  - `"background"`: Cited for contextual or literature support
  - `"supporting"`: Used as secondary or robustness-check data
- **Use Case**: Helps understand the role and importance of each dataset in the research

---

## Schema Configuration

### Default Threshold

The default confidence threshold for all fields is **0.85**. This can be customized when initializing the schema:

```python
from ai4data import DatasetSchema

# Use default threshold (0.85)
schema_builder = DatasetSchema()

# Use custom default threshold
schema_builder = DatasetSchema(threshold=0.90)
```

### Field-Specific Thresholds

You can set different thresholds for specific fields:

```python
schema_builder = DatasetSchema(threshold=0.85)
schema_builder.set_threshold("dataset_name", 0.95)  # Higher precision for names
schema_builder.set_threshold("acronym", 0.90)       # Higher precision for acronyms
```

Currently, the following fields support custom thresholds:
- `dataset_name`
- `acronym`

---

## Example Extraction

### Input Text

```text
Our analysis uses the 2022 Demographic and Health Survey (DHS) conducted by
the National Statistics Office collected for years 2010-2019 consists of demographic
and employment indicators. The DHS provides nationally representative data for women
aged 15–49, especially on health and fertility indicators. We complement the DHS with
descriptive statistics from administrative systems, but only the DHS is used in the
empirical models.
```

### Expected Output

```python
{
  'dataset_mention': [{
    'dataset_name': {
      'text': '2022 Demographic and Health Survey',
      'confidence': 0.91
    },
    'tag': 'named',
    'data_type': 'survey',
    'description': 'demographic and employment indicators',
    'acronym': {
      'text': 'DHS',
      'confidence': 0.99
    },
    'producer': {
      'text': 'National Statistics Office',
      'confidence': 0.99
    },
    'publication_year': '2022',
    'reference_year': '2010-2019',
    'reference_population': 'women aged 15–49',
    'is_used': 'true',
    'mention_context': 'primary'
  }]
}
```

---

## Multi-Dataset Extraction

The schema supports extracting **multiple dataset mentions** from a single text passage. This is common in research papers where authors use primary data supplemented with additional sources.

### Example: Multiple Datasets

#### Input Text

```text
Our analysis employs the Demographic and Health Surveys (DHS) conducted by the ICF
International in 2018, which provides comprehensive health information about women and
children. Additionally, we supplement our findings with data from the World Bank's
Poverty and Equity Database from 2015, which offers insights into poverty levels and
economic disparities.
```

#### Expected Output

```python
{
  'dataset_mention': [
    {
      'dataset_name': {
        'text': 'Demographic and Health Surveys',
        'confidence': 0.92
      },
      'tag': 'named',
      'data_type': 'survey',
      'acronym': {
        'text': 'DHS',
        'confidence': 0.98
      },
      'producer': {
        'text': 'ICF International',
        'confidence': 0.95
      },
      'publication_year': '2018',
      'reference_population': 'women and children',
      'is_used': 'true',
      'mention_context': 'primary'
    },
    {
      'dataset_name': {
        'text': "World Bank's Poverty and Equity Database",
        'confidence': 0.89
      },
      'tag': 'descriptive',
      'data_type': 'database',
      'producer': {
        'text': 'World Bank',
        'confidence': 0.96
      },
      'publication_year': '2015',
      'is_used': 'true',
      'mention_context': 'supporting'
    }
  ]
}
```

### Multi-Dataset Patterns

Common patterns in research papers:

1. **Primary + Supporting**: Main dataset supplemented with additional data
2. **Cross-Validation**: Multiple datasets for robustness checks
3. **Comparison**: Comparing different data sources
4. **Background + Primary**: Previous studies' data vs. current analysis
5. **Longitudinal**: Multiple time periods of the same survey

### Usage Context in Multi-Dataset Scenarios

- **`primary`**: The main analytical dataset (typically one per document)
- **`supporting`**: Secondary data for validation or supplementary analysis
- **`background`**: Datasets mentioned for context but not used in analysis

---

## Field Relationships and Best Practices

### Temporal Fields
- **`publication_year`** vs **`reference_year`**: Always distinguish between when data was published vs. when it was collected
- Example: Census 2020 data published in 2021:
  - `reference_year`: "2020"
  - `publication_year`: "2021"

### Classification Fields
- **`tag`** determines the specificity of the dataset mention
- **`data_type`** categorizes the nature of data collection
- **`description`** provides content-level details

### Usage Fields
- **`is_used`**: Binary indicator of analytical use
- **`mention_context`**: Granular classification of usage role
- These work together to understand dataset importance in research
