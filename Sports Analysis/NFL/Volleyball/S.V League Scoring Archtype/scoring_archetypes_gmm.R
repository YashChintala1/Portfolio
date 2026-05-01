suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(mclust)
  library(scales)
})

input_path <- "sv_league_men_2024_25_top50_players_cleaned.xlsx"
output_dir <- "outputs"
candidate_components <- 2:6
share_tolerance <- 0.01

if (!file.exists(input_path)) {
  stop("Input file not found: ", input_path, call. = FALSE)
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

required_columns <- c(
  "official_rank",
  "player_name",
  "team",
  "total_points",
  "matches",
  "sets",
  "attack_points",
  "block_points",
  "serve_points"
)

numeric_columns <- c(
  "official_rank",
  "total_points",
  "matches",
  "sets",
  "attack_points",
  "block_points",
  "serve_points"
)

feature_columns <- c(
  "points_per_set",
  "attack_share",
  "block_share",
  "serve_share"
)

read_player_data <- function(path) {
  read_excel(path, skip = 3) |>
    janitor_like_names() |>
    mutate(across(all_of(numeric_columns), as.numeric))
}

janitor_like_names <- function(df) {
  cleaned_names <- tolower(names(df))
  cleaned_names <- gsub("[^a-z0-9]+", "_", cleaned_names)
  cleaned_names <- gsub("^_|_$", "", cleaned_names)
  names(df) <- cleaned_names
  df
}

validate_player_data <- function(df) {
  missing_required <- setdiff(required_columns, names(df))
  if (length(missing_required) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_required, collapse = ", "),
      call. = FALSE
    )
  }

  invalid_rows <- df |>
    filter(
      is.na(player_name) |
        is.na(team) |
        if_any(all_of(numeric_columns), is.na)
    )

  if (nrow(invalid_rows) > 0) {
    stop("Dataset contains missing required values after import.", call. = FALSE)
  }

  if (any(df$sets <= 0)) {
    stop("All players must have sets > 0 to compute per-set rates.", call. = FALSE)
  }

  if (any(df$total_points <= 0)) {
    stop("All players must have total_points > 0 to compute scoring shares.", call. = FALSE)
  }

  component_totals <- df$attack_points + df$block_points + df$serve_points
  inconsistent_totals <- which(component_totals != df$total_points)
  if (length(inconsistent_totals) > 0) {
    stop(
      "Attack + block + serve points do not match total_points for rows: ",
      paste(inconsistent_totals, collapse = ", "),
      call. = FALSE
    )
  }
}

engineer_features <- function(df) {
  engineered <- df |>
    mutate(
      points_per_set = total_points / sets,
      attack_points_per_set = attack_points / sets,
      block_points_per_set = block_points / sets,
      serve_points_per_set = serve_points / sets,
      attack_share = attack_points / total_points,
      block_share = block_points / total_points,
      serve_share = serve_points / total_points,
      share_sum = attack_share + block_share + serve_share
    )

  bad_share_rows <- which(abs(engineered$share_sum - 1) > share_tolerance)
  if (length(bad_share_rows) > 0) {
    stop(
      "Scoring shares do not sum to approximately 1 for rows: ",
      paste(bad_share_rows, collapse = ", "),
      call. = FALSE
    )
  }

  engineered
}

fit_gmm_models <- function(feature_matrix, components) {
  fits <- setNames(vector("list", length(components)), as.character(components))
  metrics <- vector("list", length(components))

  for (i in seq_along(components)) {
    g <- components[i]
    fit <- Mclust(feature_matrix, G = g, verbose = FALSE)
    fits[[as.character(g)]] <- fit

    metrics[[i]] <- tibble(
      n_components = g,
      covariance_model = fit$modelName,
      bic = fit$bic,
      aic = (2 * fit$df) - (2 * fit$loglik),
      log_likelihood = fit$loglik,
      n_parameters = fit$df
    )
  }

  metrics <- bind_rows(metrics) |>
    arrange(desc(bic), aic)

  list(fits = fits, metrics = metrics)
}

name_archetypes <- function(summary_tbl) {
  volume_idx <- which.max(summary_tbl$points_per_set)
  block_idx <- which.max(summary_tbl$block_share)
  serve_idx <- which.max(summary_tbl$serve_share)

  labels <- rep("Balanced Scorers", nrow(summary_tbl))
  labels[volume_idx] <- "Volume Attackers"

  if (block_idx != volume_idx) {
    labels[block_idx] <- "Block-Impact Scorers"
  }

  if (serve_idx != volume_idx && serve_idx != block_idx) {
    labels[serve_idx] <- "Serve-Impact Scorers"
  }

  make.unique(labels, sep = " ")
}

player_data <- read_player_data(input_path)
validate_player_data(player_data)

player_features <- engineer_features(player_data)

scaled_features <- scale(player_features[, feature_columns])
scaled_feature_df <- as.data.frame(scaled_features)

gmm_results <- fit_gmm_models(scaled_features, candidate_components)
model_metrics <- gmm_results$metrics
best_g <- model_metrics$n_components[1]
best_fit <- gmm_results$fits[[as.character(best_g)]]

posterior_probabilities <- as.data.frame(best_fit$z)
colnames(posterior_probabilities) <- paste0("archetype_", seq_len(ncol(posterior_probabilities)), "_prob")

player_assignments <- bind_cols(player_features, posterior_probabilities) |>
  mutate(
    archetype_id = best_fit$classification,
    max_membership_probability = apply(best_fit$z, 1, max)
  )

archetype_summary <- player_assignments |>
  group_by(archetype_id) |>
  summarise(
    players = n(),
    avg_points_per_set = mean(points_per_set),
    avg_attack_share = mean(attack_share),
    avg_block_share = mean(block_share),
    avg_serve_share = mean(serve_share),
    avg_attack_points_per_set = mean(attack_points_per_set),
    avg_block_points_per_set = mean(block_points_per_set),
    avg_serve_points_per_set = mean(serve_points_per_set),
    avg_matches = mean(matches),
    avg_sets = mean(sets),
    .groups = "drop"
  ) |>
  arrange(archetype_id)

archetype_names <- name_archetypes(
  archetype_summary |>
    transmute(
      archetype_id,
      points_per_set = avg_points_per_set,
      attack_share = avg_attack_share,
      block_share = avg_block_share,
      serve_share = avg_serve_share
    )
)

archetype_summary <- archetype_summary |>
  mutate(archetype_name = archetype_names) |>
  relocate(archetype_name, .after = archetype_id)

name_lookup <- archetype_summary |>
  select(archetype_id, archetype_name)

player_assignments <- player_assignments |>
  left_join(name_lookup, by = "archetype_id") |>
  relocate(archetype_name, .after = archetype_id)

pca_fit <- prcomp(scaled_features, center = FALSE, scale. = FALSE)
pca_scores <- as.data.frame(pca_fit$x[, 1:2])
colnames(pca_scores) <- c("pc1", "pc2")

player_plot_data <- bind_cols(player_assignments, pca_scores)

archetype_heatmap <- player_assignments |>
  group_by(archetype_name) |>
  summarise(
    points_per_set = mean(points_per_set),
    attack_share = mean(attack_share),
    block_share = mean(block_share),
    serve_share = mean(serve_share),
    .groups = "drop"
  ) |>
  mutate(across(-archetype_name, ~ as.numeric(scale(.x)))) |>
  pivot_longer(
    cols = -archetype_name,
    names_to = "feature",
    values_to = "scaled_mean"
  )

archetype_scoring_mix <- player_assignments |>
  group_by(archetype_name) |>
  summarise(
    attack_share = mean(attack_share),
    block_share = mean(block_share),
    serve_share = mean(serve_share),
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = c(attack_share, block_share, serve_share),
    names_to = "scoring_source",
    values_to = "share"
  )

team_archetype_breakdown <- player_assignments |>
  dplyr::count(team, archetype_name, name = "players") |>
  arrange(team, desc(players), archetype_name)

hybrid_players <- player_assignments |>
  select(player_name, team, archetype_name, max_membership_probability, starts_with("archetype_")) |>
  arrange(max_membership_probability)

write.csv(model_metrics, file.path(output_dir, "model_selection_metrics.csv"), row.names = FALSE)
write.csv(player_assignments, file.path(output_dir, "player_archetype_assignments.csv"), row.names = FALSE)
write.csv(archetype_summary, file.path(output_dir, "archetype_feature_summary.csv"), row.names = FALSE)
write.csv(archetype_scoring_mix, file.path(output_dir, "archetype_scoring_mix.csv"), row.names = FALSE)
write.csv(team_archetype_breakdown, file.path(output_dir, "team_archetype_breakdown.csv"), row.names = FALSE)
write.csv(hybrid_players, file.path(output_dir, "hybrid_players.csv"), row.names = FALSE)

pca_plot <- ggplot(player_plot_data, aes(x = pc1, y = pc2, color = archetype_name)) +
  geom_point(size = 3, alpha = 0.9) +
  geom_text(aes(label = player_name), size = 2.8, vjust = -0.8, show.legend = FALSE) +
  labs(
    title = "SV.League Men Top-50 Scoring Archetypes",
    subtitle = paste0("Gaussian mixture model selected ", best_g, " archetypes by BIC"),
    x = "Principal Component 1",
    y = "Principal Component 2",
    color = "Archetype"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(output_dir, "pca_archetype_map.png"),
  plot = pca_plot,
  width = 11,
  height = 7,
  dpi = 300
)

heatmap_plot <- ggplot(archetype_heatmap, aes(x = feature, y = archetype_name, fill = scaled_mean)) +
  geom_tile(color = "white") +
  geom_text(aes(label = number(scaled_mean, accuracy = 0.01)), size = 3.2) +
  scale_fill_gradient2(low = "#295C77", mid = "white", high = "#B5402C", midpoint = 0) +
  labs(
    title = "Archetype Feature Heatmap",
    x = NULL,
    y = NULL,
    fill = "Scaled mean"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave(
  filename = file.path(output_dir, "archetype_profile_heatmap.png"),
  plot = heatmap_plot,
  width = 10,
  height = 6,
  dpi = 300
)

scoring_mix_plot <- ggplot(archetype_scoring_mix, aes(x = archetype_name, y = share, fill = scoring_source)) +
  geom_col() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Average Scoring Mix by Archetype",
    x = NULL,
    y = "Share of total points",
    fill = "Source"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave(
  filename = file.path(output_dir, "archetype_scoring_mix.png"),
  plot = scoring_mix_plot,
  width = 10,
  height = 6,
  dpi = 300
)

team_breakdown_plot <- ggplot(team_archetype_breakdown, aes(x = archetype_name, y = team, fill = players)) +
  geom_tile(color = "white") +
  geom_text(aes(label = players), size = 3.1) +
  scale_fill_gradient(low = "#DCEAF0", high = "#295C77") +
  labs(
    title = "Team by Archetype Breakdown",
    x = NULL,
    y = NULL,
    fill = "Players"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave(
  filename = file.path(output_dir, "team_archetype_breakdown.png"),
  plot = team_breakdown_plot,
  width = 10,
  height = 7,
  dpi = 300
)

hybrid_plot <- hybrid_players |>
  mutate(player_name = reorder(player_name, max_membership_probability))

hybrid_confidence_plot <- ggplot(hybrid_plot, aes(x = player_name, y = max_membership_probability, fill = archetype_name)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title = "Hybrid Confidence by Player",
    subtitle = "Lower top-membership probabilities indicate more mixed archetype profiles",
    x = NULL,
    y = "Top archetype probability",
    fill = "Assigned archetype"
  ) +
  theme_minimal(base_size = 11)

ggsave(
  filename = file.path(output_dir, "hybrid_confidence.png"),
  plot = hybrid_confidence_plot,
  width = 10,
  height = 12,
  dpi = 300
)

message("Scoring archetype analysis complete.")
message("Selected number of archetypes by BIC: ", best_g)
message("Outputs saved to: ", normalizePath(output_dir, winslash = "/"))
