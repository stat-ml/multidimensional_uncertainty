# multidimensional_uncertainty

This project evaluates scalar and vector-based uncertainty measures for image
classification experiments. The main idea is the VecUQ-OT pipeline from the
paper: compute several 1D uncertainty scores for each point, stack them into an
uncertainty vector, fit an aggregation map on in-distribution calibration
points, and use the resulting scalar score for OOD detection,
misclassification detection, and selective prediction.

## Data Layout

The evaluation scripts expect model logits/embeddings under:

```bash
resources/model_weights/{ind_dataset}/...
```

The 1D uncertainty scores are saved under:

```bash
resources/results_cleaned/{ind_dataset}/{measure}/{ood_dataset}/*.npz
```

Each `.npz` stores three arrays used by the evaluator:

- `ind_calib`: in-distribution calibration scores used to fit aggregators.
- `ind_test`: in-distribution test scores used for ID-side evaluation.
- `ood`: OOD scores used for OOD detection.

## Main Workflow

1. Make sure the logits/embeddings are available in `resources/model_weights`.

2. Compute all 1D uncertainty measures:

```bash
uv run bash scripts/uncertainty_script.sh
```

This calls `scripts/compute_measures_1d.py` for risk-based measures,
Mahalanobis, and GMM, and writes `.npz` files to `resources/results_cleaned`.

3. Define vector compositions in:

```bash
configs/interesting_compositions.py
```

Each composition is a list of 1D measures. The helper DSL in
`configs/config_utils.py` converts compact strings such as
`risk_bayes_logscore_outer_T=1.0` into the config dictionaries used by the
evaluation code.

4. Run the full evaluation:

```bash
uv run python scripts/full_evaluation.py \
  --entropic_target Exp \
  --entropic_sampling_method Grid \
  --entropic_scaling_type FeatureWise \
  --entropic_eps 0.5 \
  --entropic_grid_size 5 \
  --entropic_n_targets_multiplier 1 \
  --output_file ./resources/refactored/results.csv \
  --verbose
```

By default this evaluates:

- every individual 1D measure;
- every configured composition with EntropicOT (`uncertainty_type=EntropicOT`);
- every configured composition with the additive baseline
  (`uncertainty_type=Additive`).

Use `--skip_additive_baseline` to disable the naive summation baseline.

5. Build article tables from the full evaluation CSV:

```bash
uv run python scripts/build_paper_tables.py \
  --input_csv ./resources/refactored/results.csv \
  --output_dir ./resources/paper_tables
```

This writes:

- `all_tasks_mean.csv` and `all_tasks_std.csv`;
- OOD, misclassification, and selective prediction tables under
  `resources/paper_tables/problem_tables`;
- LaTeX tables under `resources/paper_tables/latex`;
- per-composition tables with components plus EntropicOT and additive
  baselines under `resources/paper_tables/composition_tables`;
- `average_ranks.csv`, `measure_summary.csv`, `pareto_summary.csv`, and
  `article_pareto_table.csv`.

`pareto_summary.csv` counts EntropicOT and additive aggregations on the same
component-wise Pareto-front criterion.
`article_pareto_table.csv` and `latex/article_pareto_table.tex` rebuild the
right-hand Table 2-style Pareto-front table for
`COMPOSITE EAT LOGSCORE OUTER OUTER + M`, comparing Ours, Additive, and the
individual components in the same Pareto-front calculation.

There is also a thin notebook wrapper at `notebooks/paper_tables.ipynb` if you
want to inspect the generated tables interactively.

6. Find cases where one aggregation clearly wins and the others trail:

```bash
uv run python scripts/find_aggregation_contrasts.py \
  --input_csv ./resources/refactored/results.csv \
  --output_dir ./resources/paper_tables/aggregation_contrasts \
  --min_gap 0.05 \
  --min_broken 1
```

This compares EntropicOT and additive aggregation for each composition and task
row. A row is selected when the winner beats at least `--min_broken` other
aggregation methods by at least `--min_gap`. The script writes
`contrast_cases.csv`, summary CSVs, and small selected tables under
`resources/paper_tables/aggregation_contrasts`. Use
`notebooks/aggregation_contrast_insights.ipynb` to inspect where either
aggregation tends to break.

For a global winner-rate view of `Ours` and additive aggregation, use
`notebooks/aggregation_winner_barplots.ipynb`. It writes five barplots to
`resources/paper_tables/aggregation_winner_barplots`: overall, OOD detection,
misclassification detection, selective prediction, and LLM selective generation.

## Aggregators

`mdu/unc/entropic_ot.py` contains `EntropicOTOrdering`. It fits an entropic OT
map on calibration uncertainty vectors and returns the norm of the barycentric
rank image as the final scalar score.

`mdu/unc/additive_baseline.py` contains `AdditiveUncertaintyOrdering`. It is the
most naive baseline: no scaling, no learned weights, just the raw sum of all
component uncertainty scores in the vector.

All aggregators expose the same minimal API:

```python
model.fit(scores_cal)
scores = model.predict(scores_test)
```

where `scores_cal` and `scores_test` have shape `(n_samples, n_measures)`.

## Toy Experiment

`scripts/main_toy.py` trains a small shallow network on synthetic 2D blobs,
computes the configured 1D uncertainty measures on a grid, and visualizes:

- the individual 1D component scores;
- VecUQ-OT (`multidim_scores`);
- additive summation (`additive_scores`).

The plots are written to `resources/pics`.

## Tests

The current test suite is intentionally small and fast:

```bash
uv run python -m unittest discover
```

It checks:

- additive baseline summation behavior and shape validation;
- 1D EntropicOT ordering behavior against rank/CDF-like expectations;
- evaluator smoke tests on fake `.npz`-shaped inputs for EntropicOT and
  additive baselines.

For a quick syntax check of the main scripts:

```bash
uv run python -m py_compile \
  scripts/full_evaluation.py \
  scripts/compute_measures_1d.py \
  scripts/compose_multidimensional_scores.py \
  scripts/build_paper_tables.py \
  scripts/find_aggregation_contrasts.py \
  scripts/main_toy.py \
  scripts/main_images_ood.py \
  scripts/main_images_mis.py
```
