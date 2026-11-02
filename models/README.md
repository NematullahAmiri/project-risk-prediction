# Model artifacts

The notebook currently exports `project_risk_model.pkl`, containing the XGBoost estimator only.

This is not yet a complete inference artifact: preprocessing steps and the original target-label mapping are not bundled with it. Do not use this file as a standalone production prediction service.

After correcting the data leakage and validating the workflow, save the complete fitted preprocessing-plus-estimator pipeline (and required label mapping) using a documented, versioned artifact format. Only commit model files if their size, licensing and security implications have been reviewed.
