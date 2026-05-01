# Academic Project README

## Academic Integrity Notice

This project was completed as part of a university course and is shared **strictly for viewing, learning, and portfolio/reference purposes only**.

**Do not copy, reproduce, submit, or redistribute this work as your own.**

Any reuse of this project, in whole or in part, for coursework, assignments, assessments, or academic submission may constitute **plagiarism or academic misconduct**.

By accessing this project, you acknowledge that:

- This work is not provided as a template for resubmission or reuse.
- You are solely responsible for how you use any material shown here.
- I am **not responsible** for any consequences, penalties, misconduct findings, grade sanctions, or disciplinary action resulting from plagiarism, copying, or unauthorized reproduction of this work.

## Project Overview

This academic project focuses on building predictive models in **R** to estimate short-term rental listing prices using structured listing, host, review, and location-related features.

The project includes:

- Data preprocessing and cleaning
- Regression-based modeling
- Tree-based and ensemble modeling
- Model comparison using cross-validation metrics
- Final prediction generation on a held-out test dataset

Based on the modeling workflow in the R Markdown analysis, **Random Forest** was identified as the strongest overall model among the approaches evaluated.

## Project Files

- `code_CHINTALA_YASHASVI_YYC3.Rmd` - Main R Markdown source file containing preprocessing, modeling, evaluation, and prediction export logic
- `code_CHINTALA_YASHASVI_YYC3.pdf` - Rendered PDF version of the analysis/code report
- `df_train.csv` - Training dataset used to build and evaluate models
- `df-test-1.csv` - Test dataset used for final prediction generation
- `testing_predictions_CHINTALA_YASHASVI_YYC3.csv` - Final output file containing predicted prices for the test set
- `final_report_CHINTALA_YASHASVI_YYC3.docx` - Final written report
- `technical_report_CHINTALA_YASHASVI_YYC3.docx` - Technical report documenting methods and modeling details

## Methods Used

The project workflow includes:

- Text and categorical field cleaning
- Numeric validation and filtering
- Missing/invalid value handling
- Polynomial regression
- Ridge regression
- Lasso regression
- Random Forest
- Bagging
- Gradient Boosting
- Cross-validation-based model comparison

## Tools and Libraries

The analysis is written in **R / R Markdown** and uses packages such as:

- `dplyr`
- `stringr`
- `caret`
- `ggplot2`
- `randomForest`
- `gbm`

## Reproducibility Notes

To rerun the project, the required datasets should be available in the same working directory as the `.Rmd` file, and the required R packages must be installed before knitting or executing the analysis.

## Ownership

Author: **Yashasvi Chintala**

All original work in this project remains the intellectual and academic work product of the author unless otherwise stated.
