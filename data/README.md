# Data documentation

## Expected input

The supplied notebook expects a CSV file named `Project_risk_raw_dataset.csv`.

The notebook reports 4,000 observations and 51 original columns. It removes `Project_ID` before model development and uses `Risk_Level` as the prediction target.

## Information still required

Before publishing or relying on the dataset, document:

- Original dataset title and source URL or provider.
- Dataset owner and licence / redistribution permission.
- Collection period and geographic or organizational coverage, if known.
- Definition of one observation (what constitutes a project record).
- Data dictionary for all 51 original columns.
- Definition and original labels of `Risk_Level`.
- Class counts and class proportions.
- Missing-value counts and any known data-quality limitations.
- Whether any variables would be unavailable at the intended prediction time.

## Privacy and licensing

Do not upload confidential, personal, proprietary or restricted project data. If redistribution is not permitted, keep the data out of Git and explain how an authorised user can obtain it.

The dataset was not included with the notebook provided for this review, so its provenance and licence have not been verified.
