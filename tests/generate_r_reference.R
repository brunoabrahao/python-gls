#!/usr/bin/env Rscript
# generate_r_reference.R
#
# Generates reference data and fitted results from R's nlme::gls() for
# 12 scenarios.  Output: tests/fixtures/r_reference.json
#
# Usage:
#   Rscript tests/generate_r_reference.R

library(nlme)
library(jsonlite)

# ---------------------------------------------------------------------------
# Helper: extract common results from a gls fit
# ---------------------------------------------------------------------------
extract_results <- function(fit, has_cor = FALSE, has_var = FALSE) {
  s <- summary(fit)
  tt <- s$tTable

  # Ensure std_errors is always a named list (single-row tTable drops names)
  se_vec <- tt[, "Std.Error"]
  if (is.null(names(se_vec))) {
    names(se_vec) <- rownames(tt)
  }

  res <- list(
    coefficients = as.list(coef(fit)),
    std_errors   = as.list(se_vec),
    sigma        = fit$sigma,
    loglik       = as.numeric(logLik(fit)),
    aic          = AIC(fit),
    bic          = BIC(fit),
    nobs         = nrow(fit$data)
  )

  if (has_cor) {
    cs <- fit$modelStruct$corStruct
    res$correlation_params <- as.list(coef(cs, unconstrained = FALSE))
  } else {
    res$correlation_params <- NULL
  }

  if (has_var) {
    vs <- fit$modelStruct$varStruct
    res$variance_params <- as.list(coef(vs, unconstrained = FALSE))
  } else {
    res$variance_params <- NULL
  }

  res
}

scenarios <- list()

# ===========================================================================
# Scenario 1: OLS baseline (no correlation, no variance, REML)
# ===========================================================================
{
  set.seed(1)
  n <- 200
  x1 <- rnorm(n)
  y  <- 2.0 + 1.5 * x1 + rnorm(n)
  subject <- seq_len(n)            # every obs is its own group
  df <- data.frame(y = y, x1 = x1, subject = subject)

  fit <- gls(y ~ x1, data = df, method = "REML")

  scenarios[["ols_baseline"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "none", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit)
  )
}

# ===========================================================================
# Scenario 2: AR(1) moderate phi, REML
# ===========================================================================
{
  set.seed(2)
  n_subj <- 50; n_time <- 6
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  x1      <- rnorm(n_subj * n_time)

  phi_true <- 0.5
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    e <- numeric(n_time)
    e[1] <- rnorm(1)
    for (t in 2:n_time) {
      e[t] <- phi_true * e[t - 1] + sqrt(1 - phi_true^2) * rnorm(1)
    }
    y[idx] <- 1.0 + 2.0 * x1[idx] + e
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time)

  fit <- gls(y ~ x1, data = df,
             correlation = corAR1(form = ~time | subject),
             method = "REML")

  scenarios[["ar1_moderate_reml"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "AR1", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 3: AR(1) moderate phi, ML
# ===========================================================================
{
  set.seed(3)
  n_subj <- 50; n_time <- 6
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  x1      <- rnorm(n_subj * n_time)

  phi_true <- 0.5
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    e <- numeric(n_time)
    e[1] <- rnorm(1)
    for (t in 2:n_time) {
      e[t] <- phi_true * e[t - 1] + sqrt(1 - phi_true^2) * rnorm(1)
    }
    y[idx] <- 1.0 + 2.0 * x1[idx] + e
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time)

  fit <- gls(y ~ x1, data = df,
             correlation = corAR1(form = ~time | subject),
             method = "ML")

  scenarios[["ar1_moderate_ml"]] <- list(
    data   = as.list(df),
    config = list(method = "ML", correlation = "AR1", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 4: AR(1) high phi, REML
# ===========================================================================
{
  set.seed(4)
  n_subj <- 60; n_time <- 8
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  x1      <- rnorm(n_subj * n_time)

  phi_true <- 0.8
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    e <- numeric(n_time)
    e[1] <- rnorm(1)
    for (t in 2:n_time) {
      e[t] <- phi_true * e[t - 1] + sqrt(1 - phi_true^2) * rnorm(1)
    }
    y[idx] <- 1.0 + 2.0 * x1[idx] + e
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time)

  fit <- gls(y ~ x1, data = df,
             correlation = corAR1(form = ~time | subject),
             method = "REML")

  scenarios[["ar1_high_reml"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "AR1", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 5: AR(1) negative phi, REML
# ===========================================================================
{
  set.seed(5)
  n_subj <- 50; n_time <- 5
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  x1      <- rnorm(n_subj * n_time)

  phi_true <- -0.4
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    e <- numeric(n_time)
    e[1] <- rnorm(1)
    for (t in 2:n_time) {
      e[t] <- phi_true * e[t - 1] + sqrt(1 - phi_true^2) * rnorm(1)
    }
    y[idx] <- 1.0 + 2.0 * x1[idx] + e
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time)

  fit <- gls(y ~ x1, data = df,
             correlation = corAR1(form = ~time | subject),
             method = "REML")

  scenarios[["ar1_negative_reml"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "AR1", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 6: CompSymm, REML
# ===========================================================================
{
  set.seed(6)
  n_subj <- 50; n_time <- 5
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  x1      <- rnorm(n_subj * n_time)

  rho_true <- 0.4
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    u <- rnorm(1) * sqrt(rho_true)
    e <- rnorm(n_time) * sqrt(1 - rho_true)
    y[idx] <- 2.0 + 1.0 * x1[idx] + u + e
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time)

  fit <- gls(y ~ x1, data = df,
             correlation = corCompSymm(form = ~time | subject),
             method = "REML")

  scenarios[["compsymm_reml"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "CompSymm", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 7: CompSymm, ML
# ===========================================================================
{
  set.seed(7)
  n_subj <- 50; n_time <- 5
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  x1      <- rnorm(n_subj * n_time)

  rho_true <- 0.4
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    u <- rnorm(1) * sqrt(rho_true)
    e <- rnorm(n_time) * sqrt(1 - rho_true)
    y[idx] <- 2.0 + 1.0 * x1[idx] + u + e
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time)

  fit <- gls(y ~ x1, data = df,
             correlation = corCompSymm(form = ~time | subject),
             method = "ML")

  scenarios[["compsymm_ml"]] <- list(
    data   = as.list(df),
    config = list(method = "ML", correlation = "CompSymm", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 8: VarIdent (2 groups), REML
# ===========================================================================
{
  set.seed(8)
  n_subj <- 60; n_time <- 5
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  group   <- ifelse(subject <= 30, "A", "B")
  x1      <- rnorm(n_subj * n_time)

  sigma_a <- 1.0; sigma_b <- 2.5
  sigma_vec <- ifelse(group == "A", sigma_a, sigma_b)
  y <- 3.0 + 0.5 * x1 + rnorm(n_subj * n_time) * sigma_vec

  df <- data.frame(y = y, x1 = x1, subject = subject, time = time, group = group)

  fit <- gls(y ~ x1, data = df,
             weights = varIdent(form = ~1 | group),
             method = "REML")

  scenarios[["varident_reml"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "none", variance = "VarIdent",
                  variance_group = "group",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_var = TRUE)
  )
}

# ===========================================================================
# Scenario 9: AR(1) + VarIdent, REML
# ===========================================================================
{
  set.seed(9)
  n_subj <- 60; n_time <- 5
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  group   <- ifelse(subject <= 30, "A", "B")
  x1      <- rnorm(n_subj * n_time)

  phi_true <- 0.5
  sigma_a <- 1.0; sigma_b <- 2.0
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    sig <- if (s <= 30) sigma_a else sigma_b
    e <- numeric(n_time)
    e[1] <- rnorm(1) * sig
    for (t in 2:n_time) {
      e[t] <- phi_true * e[t - 1] + sqrt(1 - phi_true^2) * rnorm(1) * sig
    }
    y[idx] <- 1.5 + 1.0 * x1[idx] + e
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time, group = group)

  fit <- gls(y ~ x1, data = df,
             correlation = corAR1(form = ~time | subject),
             weights   = varIdent(form = ~1 | group),
             method = "REML")

  scenarios[["ar1_varident_reml"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "AR1", variance = "VarIdent",
                  variance_group = "group",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE, has_var = TRUE)
  )
}

# ===========================================================================
# Scenario 10: Multiple predictors + AR(1), REML
# ===========================================================================
{
  set.seed(10)
  n_subj <- 50; n_time <- 6
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)
  x1      <- rnorm(n_subj * n_time)
  x2      <- rnorm(n_subj * n_time)
  x3      <- rnorm(n_subj * n_time)

  phi_true <- 0.4
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    e <- numeric(n_time)
    e[1] <- rnorm(1)
    for (t in 2:n_time) {
      e[t] <- phi_true * e[t - 1] + sqrt(1 - phi_true^2) * rnorm(1)
    }
    y[idx] <- 0.5 + 1.0 * x1[idx] - 0.5 * x2[idx] + 2.0 * x3[idx] + e
  }
  df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3,
                   subject = subject, time = time)

  fit <- gls(y ~ x1 + x2 + x3, data = df,
             correlation = corAR1(form = ~time | subject),
             method = "REML")

  scenarios[["multi_predictor_ar1"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "AR1", variance = "none",
                  formula = "y ~ x1 + x2 + x3"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 11: Unbalanced panels + AR(1), REML
# ===========================================================================
{
  set.seed(11)
  n_subj <- 50
  # Each subject has between 3 and 8 observations
  group_sizes <- sample(3:8, n_subj, replace = TRUE)
  N <- sum(group_sizes)

  subject <- rep(1:n_subj, times = group_sizes)
  time <- unlist(lapply(group_sizes, function(g) 1:g))
  x1 <- rnorm(N)

  phi_true <- 0.5
  y <- numeric(N)
  offset <- 0
  for (s in 1:n_subj) {
    g <- group_sizes[s]
    idx <- (offset + 1):(offset + g)
    e <- numeric(g)
    e[1] <- rnorm(1)
    for (t in 2:g) {
      e[t] <- phi_true * e[t - 1] + sqrt(1 - phi_true^2) * rnorm(1)
    }
    y[idx] <- 1.0 + 1.5 * x1[idx] + e
    offset <- offset + g
  }
  df <- data.frame(y = y, x1 = x1, subject = subject, time = time)

  fit <- gls(y ~ x1, data = df,
             correlation = corAR1(form = ~time | subject),
             method = "REML")

  scenarios[["unbalanced_ar1"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "AR1", variance = "none",
                  formula = "y ~ x1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ===========================================================================
# Scenario 12: Intercept-only + CompSymm, REML
# ===========================================================================
{
  set.seed(12)
  n_subj <- 40; n_time <- 5
  subject <- rep(1:n_subj, each = n_time)
  time    <- rep(1:n_time, times = n_subj)

  rho_true <- 0.5
  y <- numeric(n_subj * n_time)
  for (s in 1:n_subj) {
    idx <- ((s - 1) * n_time + 1):(s * n_time)
    u <- rnorm(1) * sqrt(rho_true)
    e <- rnorm(n_time) * sqrt(1 - rho_true)
    y[idx] <- 5.0 + u + e
  }
  df <- data.frame(y = y, subject = subject, time = time)

  fit <- gls(y ~ 1, data = df,
             correlation = corCompSymm(form = ~time | subject),
             method = "REML")

  scenarios[["intercept_only_cs"]] <- list(
    data   = as.list(df),
    config = list(method = "REML", correlation = "CompSymm", variance = "none",
                  formula = "y ~ 1"),
    results = extract_results(fit, has_cor = TRUE)
  )
}

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------
# Determine script directory robustly
args <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_arg) > 0) {
  script_dir <- dirname(normalizePath(script_arg))
} else {
  script_dir <- getwd()
}
out_dir <- file.path(script_dir, "fixtures")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, "r_reference.json")

write_json(scenarios, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 15)
cat("Wrote", length(scenarios), "scenarios to", out_path, "\n")
