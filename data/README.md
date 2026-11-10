# Dataset Information

## Restaurant Review Dataset

This project uses a restaurant review dataset for aspect-based sentiment analysis (ABSA).

**Dataset file:** `restaurant_reviews.xlsx`

**Total records:** 4,827

### Dataset Columns

| Column | Description |
|---|---|
| `text` | Customer review text |
| `span` | Restaurant aspect or annotated text span |
| `label` | Sentiment annotation |
| `ordinal` | Additional annotation information |

### Dataset Usage

The dataset is used to analyse customer opinions about restaurant experiences, including food, service, ambience and other aspects.

The Python implementation loads the Excel dataset using Pandas and performs preprocessing, feature extraction, model training and evaluation.

### Availability

The Excel dataset is not currently included in this public repository while its original source and redistribution permissions are being confirmed.

To reproduce the analysis, place an appropriately licensed copy of `restaurant_reviews.xlsx` in the repository's root directory, alongside `restaurant_review_analysis.py`.

The complete analysis has not yet been rerun and verified using the recovered dataset.
