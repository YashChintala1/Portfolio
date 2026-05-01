# Scoring Archetypes in R

This project builds Gaussian Mixture Model (GMM) scoring archetypes for the top 50 scorers in the 2024-25 SV.League Men season using the workbook `sv_league_men_2024_25_top50_players_cleaned.xlsx`.

## What the script does

- reads the Excel workbook while skipping the metadata rows
- engineers per-set and scoring-share features
- validates the scoring-share math and missing values
- standardizes the archetype features
- fits GMMs for 2 through 6 archetypes
- selects the best model by BIC
- assigns each player to an archetype and saves posterior probabilities
- creates summary tables and visualizations

## Files

- `scripts/scoring_archetypes_gmm.R`: main analysis script
- `outputs/`: generated tables and charts

## Required R packages

Install these once in R:

```r
install.packages(c(
  "readxl",
  "dplyr",
  "tidyr",
  "ggplot2",
  "mclust",
  "scales"
))
```

## Run

From this project folder, run:

```r
source("scripts/scoring_archetypes_gmm.R")
```

Or from a terminal:

```bash
Rscript scripts/scoring_archetypes_gmm.R
```

## Main outputs

- `outputs/model_selection_metrics.csv`
- `outputs/player_archetype_assignments.csv`
- `outputs/archetype_feature_summary.csv`
- `outputs/archetype_scoring_mix.csv`
- `outputs/team_archetype_breakdown.csv`
- `outputs/hybrid_players.csv`
- `outputs/pca_archetype_map.png`
- `outputs/archetype_profile_heatmap.png`
- `outputs/archetype_scoring_mix.png`
- `outputs/team_archetype_breakdown.png`
- `outputs/hybrid_confidence.png`

## Notes

- The analysis is scoped to the top 50 scorers only, so the archetypes describe high-output scorers rather than the full league.
- The current machine did not have `Rscript` available on `PATH`, so the script was built but not executed here.
