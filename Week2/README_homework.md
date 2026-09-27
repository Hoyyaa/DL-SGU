# Homework package guide

## Files to submit

- `house_price.ipynb`: concise CRISP-DM walkthrough, inline EDA chart, model results, and submission preview.
- `house_price_train.py`: complete source for EDA, preprocessing, CV, scikit-learn models, PyTorch MLP, model selection, prediction, and automated checks.
- `report_crisp_dm.md`: full report draft with a cover page, work allocation table, methodology, results, limitations, and references.
- `requirements.txt`: package versions used for the completed run.
- `model_comparison.csv`, `oof_predictions.csv`, `run_summary.json`: validation results and reproducibility evidence.
- `submission.csv`: checked Kaggle upload file.
- `house_price_mlp.pt`, `house_price_preprocessor.joblib`: saved model and preprocessing artifacts.
- `figures/eda_target_and_living_area.png`: EDA figure used by the report.

The older `nn_model_pytorch.pth` is from the earlier notebook and is not used by this workflow.

## Reproduce

From this folder, install dependencies and run:

```bash
python -m pip install -r requirements.txt
python house_price_train.py
```

Or open `house_price.ipynb` and run all cells. It calls the same `run_workflow` function and displays the chart inline; the full implementation is in `house_price_train.py`.

## Before submission

1. Replace all `[bracketed fields]` in `report_crisp_dm.md` with the correct course, lecturer, team names, student IDs, class, place/date, and actual work division.
2. Check that the contribution table matches what each person did.
3. Run the workflow again with the final code; check `run_summary.json`, the `model_comparison.csv` metrics, and the success checks at the end of execution.
4. Submit `submission.csv` to the House Prices Kaggle competition and add the real leaderboard score to the report. Do not call OOF RMSLE the Kaggle score.
5. Export the report to the format required by the lecturer (for example, PDF) and submit the notebook/source and output artifacts requested by the course.
