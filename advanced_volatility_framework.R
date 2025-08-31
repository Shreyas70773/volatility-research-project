# #############################################################################
#
#    Advanced Volatility Modeling: Windows-Compatible Robust Framework
#    FIXED VERSION - Resolves multiple errors and improves robustness
#
# This script integrates advanced modeling with robust engineering practices,
# including pre-run system checks, resource monitoring, and checkpointing
# for long-running, parallelized analyses.
#
# Author: Shreyas Sunil
# Date: 31-08-2025
# Fixed: Target variable creation, data alignment, VaRTest implementation,
#        and summary generation errors.
#
# #############################################################################

# --- 0. Enhanced Setup, Configuration, and Diagnostics ---

# --- 0.1: Windows-Compatible Package Installation ---
# Core packages that are essential and usually work well on Windows
core_packages <- c("quantmod", "rugarch", "dplyr", "zoo", "ggplot2", 
                   "PerformanceAnalytics", "parallel", "doParallel", "curl", "reshape2")

# Optional packages that might fail - we'll handle gracefully
optional_packages <- c("rmgarch", "xgboost", "devtools")

install_with_retry <- function(packages, max_retries = 2) {
  success <- character()
  failed <- character()
  
  for (pkg in packages) {
    cat(paste("Installing", pkg, "...\n"))
    
    for (attempt in 1:max_retries) {
      result <- tryCatch({
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
          install.packages(pkg, repos = "https://cran.rstudio.com/", 
                           dependencies = TRUE, type = "binary")
          library(pkg, character.only = TRUE, quietly = TRUE)
        }
        TRUE
      }, error = function(e) {
        cat(paste("Attempt", attempt, "failed for", pkg, ":", e$message, "\n"))
        FALSE
      })
      
      if (result) {
        success <- c(success, pkg)
        break
      }
      
      if (attempt == max_retries) {
        failed <- c(failed, pkg)
      }
    }
  }
  
  list(success = success, failed = failed)
}

# Install core packages first
cat("=== INSTALLING CORE PACKAGES ===\n")
core_result <- install_with_retry(core_packages)

# Install optional packages
cat("\n=== INSTALLING OPTIONAL PACKAGES ===\n")
optional_result <- install_with_retry(optional_packages)

# Report installation results
cat("\n=== INSTALLATION SUMMARY ===\n")
cat("Successfully installed:", paste(c(core_result$success, optional_result$success), collapse = ", "), "\n")
if (length(c(core_result$failed, optional_result$failed)) > 0) {
  cat("Failed to install:", paste(c(core_result$failed, optional_result$failed), collapse = ", "), "\n")
}

# Check which advanced features are available
HAS_RMGARCH <- "rmgarch" %in% c(core_result$success, optional_result$success)
HAS_XGBOOST <- "xgboost" %in% c(core_result$success, optional_result$success)
HAS_DEVTOOLS <- "devtools" %in% c(core_result$success, optional_result$success)


cat("\nAdvanced features available:\n")
cat("- Multivariate GARCH (rmgarch):", ifelse(HAS_RMGARCH, "YES", "NO"), "\n")
cat("- Machine Learning (xgboost):", ifelse(HAS_XGBOOST, "YES", "NO"), "\n")
cat("- Development tools (devtools):", ifelse(HAS_DEVTOOLS, "YES", "NO"), "\n")

# --- 0.2: Define Constants ---
INITIAL_TRAIN_SIZE <- 1500
REFIT_EVERY <- 50
BACKTEST_WINDOW <- 1000
TEST_LENGTH <- 500
VAR_ALPHA <- 0.01

# --- 0.3: Define Diagnostic and Helper Functions ---
timestamp_message <- function(msg) {
  cat(paste0("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ", msg, "\n"))
}

monitor_resources <- function() {
  mem_used <- gc()[2, 2]
  timestamp_message(paste("Memory used:", round(mem_used, 1), "MB"))
}

save_checkpoint <- function(section_name, objects_to_save) {
  checkpoint_file <- paste0("checkpoint_", section_name, ".RData")
  tryCatch({
    save(list = objects_to_save, file = checkpoint_file, envir = parent.frame())
    timestamp_message(paste("Checkpoint saved:", checkpoint_file))
  }, error = function(e) {
    timestamp_message(paste("Failed to save checkpoint:", e$message))
  })
}

validate_data <- function(data, name) {
  original_nrow <- nrow(data)
  
  # Remove NAs
  data <- na.omit(data)
  
  # Check for infinite values
  if (is.xts(data) || is.data.frame(data)) {
    n_inf <- sum(is.infinite(as.matrix(data)))
  } else {
    n_inf <- sum(is.infinite(data))
  }
  
  if (n_inf > 0) {
    timestamp_message(paste("WARNING in", name, "- Infinite values found:", n_inf))
    if (is.xts(data) || is.data.frame(data)) {
      data[is.infinite(as.matrix(data))] <- NA
    } else {
      data[is.infinite(data)] <- NA
    }
    data <- na.omit(data)
  }
  
  final_nrow <- nrow(data)
  if (final_nrow < original_nrow) {
    timestamp_message(paste("Data validation for", name, "- Removed", 
                            original_nrow - final_nrow, "invalid observations"))
  }
  timestamp_message(paste("Data validation for", name, "- Final observations:", final_nrow))
  return(data)
}

# --- 0.4: Pre-Run System Checks ---
pre_run_check <- function() {
  cat("\n=== PRE-RUN SYSTEM CHECK ===\n")
  
  # Check internet connection
  internet_ok <- tryCatch({
    if (requireNamespace("curl", quietly = TRUE)) {
      curl::has_internet()
    } else {
      # Fallback internet check
      !is.null(suppressWarnings(readLines("https://www.google.com", n = 1, warn = FALSE)))
    }
  }, error = function(e) FALSE)
  
  if (!internet_ok) {
    cat("ERROR: No internet connection.\n")
    return(FALSE)
  } else {
    cat("✓ Internet connection OK\n")
  }
  
  # Check write permissions
  write_ok <- tryCatch({
    writeLines("test", "write_test.tmp")
    unlink("write_test.tmp")
    cat("✓ Write permissions OK\n")
    TRUE
  }, error = function(e) {
    cat("ERROR: No write permissions.\n")
    FALSE
  })
  
  if (!write_ok) return(FALSE)
  
  # Check available memory
  mem_info <- gc()
  cat("✓ Available memory:", round(mem_info[2, 2], 1), "MB\n")
  
  cat("✓ System checks passed.\n")
  return(TRUE)
}

if (!pre_run_check()) {
  stop("Pre-run checks failed. Please resolve issues before running.")
}

# --- 0.5: Configure Parallel Processing ---
n_cores <- max(1, parallel::detectCores() - 1)
if (requireNamespace("doParallel", quietly = TRUE) && HAS_XGBOOST) {
  registerDoParallel(cores = n_cores)
  timestamp_message(paste("Parallel processing enabled with", n_cores, "cores."))
} else {
  timestamp_message("Parallel processing not available - running sequentially")
  n_cores <- 1
}


# #############################################################################
# SECTION 1: MULTIVARIATE CGARCH WITH PORTFOLIO APPLICATIONS
# #############################################################################
timestamp_message("Starting Section 1: Portfolio Analysis...")

if (HAS_RMGARCH) {
  timestamp_message("Running advanced multivariate CGARCH analysis")
  
  # --- 1.1: Extended Portfolio Data ---
  tickers <- c("SPY", "QQQ", "TLT", "GLD")
  
  # Download with error handling
  download_success <- tryCatch({
    getSymbols(tickers, src = "yahoo", from = "2015-01-01", to = Sys.Date())
    TRUE
  }, error = function(e) {
    timestamp_message(paste("Error downloading data:", e$message))
    FALSE
  })
  
  if (!download_success) {
    timestamp_message("Failed to download portfolio data - skipping Section 1")
  } else {
    portfolio_returns <- do.call(merge, lapply(tickers, function(t) {
      dailyReturn(get(t), type = "log") * 100
    }))
    colnames(portfolio_returns) <- tickers
    portfolio_returns <- validate_data(portfolio_returns, "Portfolio Returns")
    
    # Check if we have sufficient data
    if (nrow(portfolio_returns) < 1000) {
      timestamp_message("Insufficient portfolio data for DCC analysis")
    } else {
      # --- 1.2: Advanced DCC-CGARCH Implementation ---
      univariate_spec <- ugarchspec(
        variance.model = list(model = "csGARCH"), 
        mean.model = list(armaOrder = c(1, 1)), 
        distribution.model = "std"
      )
      
      multivariate_spec <- multispec(replicate(ncol(portfolio_returns), univariate_spec))
      dcc_spec <- dccspec(uspec = multivariate_spec, dccOrder = c(1, 1), distribution = "mvt")
      
      recent_data <- tail(portfolio_returns, 1000)
      dcc_model <- tryCatch({
        dccfit(dcc_spec, data = recent_data, solver = "solnp")
      }, error = function(e) {
        timestamp_message(paste("DCC model fit failed:", e$message))
        NULL
      })
      
      if (!is.null(dcc_model)) {
        # --- 1.3: Analysis & Visualization ---
        R_t <- rcor(dcc_model)
        cor_spy_qqq <- R_t["SPY", "QQQ", ]
        cor_spy_tlt <- R_t["SPY", "TLT", ]
        
        plot_data <- xts(
          data.frame(SPY_QQQ = cor_spy_qqq, SPY_TLT = cor_spy_tlt), 
          order.by = index(recent_data)
        )
        
        print(plot(plot_data, main = "Dynamic Conditional Correlations (DCC-CGARCH)", 
                   legend.loc = "topright"))
        
        # --- 1.4: Checkpoint ---
        save_checkpoint("section1_dcc", c("dcc_model", "plot_data"))
        timestamp_message("DCC-CGARCH model successfully fitted and analyzed")
      }
    }
  }
} else {
  timestamp_message("rmgarch not available - running simplified portfolio correlation analysis")
  
  # Simplified correlation analysis without rmgarch
  tickers <- c("SPY", "QQQ", "TLT", "GLD")
  tryCatch({
    getSymbols(tickers, src = "yahoo", from = "2020-01-01", to = Sys.Date())
    portfolio_returns <- do.call(merge, lapply(tickers, function(t) {
      dailyReturn(get(t), type = "log") * 100
    }))
    colnames(portfolio_returns) <- tickers
    portfolio_returns <- validate_data(portfolio_returns, "Portfolio Returns")
    
    # Rolling correlation analysis
    rolling_cor <- rollapply(portfolio_returns, 60, function(x) cor(x)[1,2], align = "right")
    plot(rolling_cor, main = "Rolling 60-Day Correlation: SPY vs QQQ")
    
    save_checkpoint("section1_simple", c("portfolio_returns", "rolling_cor"))
    timestamp_message("Simplified portfolio analysis completed")
  }, error = function(e) {
    timestamp_message(paste("Portfolio analysis failed:", e$message))
  })
}

monitor_resources()
timestamp_message("Section 1 Complete.")


# #############################################################################
# SECTION 2: ADVANCED ML INTEGRATION WITH ROLLING BACKTEST
# #############################################################################
timestamp_message("Starting Section 2: ML-GARCH Framework...")

if (HAS_XGBOOST) {
  timestamp_message("Running ML-enhanced GARCH analysis")
  
  # --- 2.1: Enhanced Feature Engineering ---
  download_sp500 <- tryCatch({
    getSymbols("^GSPC", src = "yahoo", from = "2010-01-01", to = Sys.Date())
    TRUE
  }, error = function(e) {
    timestamp_message(paste("Error downloading S&P 500 data:", e$message))
    FALSE
  })
  
  if (download_sp500) {
    returns <- dailyReturn(GSPC, type = "log") * 100
    returns <- returns[-1, ]  # Remove first NA
    colnames(returns) <- "returns"
    
    # --- FIXED FUNCTION TO PREVENT ERRORS ---
    create_ml_features <- function(returns_data) {
      features <- returns_data
      features$sq_returns <- features$returns^2
      
      # Create lagged volatility features
      for (i in 1:5) {
        lag_col <- lag(features$sq_returns, i)
        colnames(lag_col) <- paste0("lag_vol_", i)
        features <- merge(features, lag_col, join = "left")
      }
      
      # Moving average volatility
      features$ma_vol_20 <- rollmean(features$sq_returns, 20, fill = NA, align = "right")
      
      # Recommendation: Additional features could be added here
      # e.g., market sentiment, macro-economic data, or regime-switching indicators
      
      # FIXED: Target variable creation using proper forward-looking approach
      # Calculate realized volatility over next 20 days
      future_vol <- rollapply(features$returns, 20, function(x) sd(x, na.rm = TRUE) * sqrt(252), 
                              align = "left", fill = NA)
      
      # Create target by shifting the future volatility series
      # This gives us forward-looking volatility as our prediction target
      n_obs <- nrow(features)
      target_values <- rep(NA, n_obs)
      
      # Fill in the target values where we have future data
      for (i in 1:(n_obs - 20)) {
        if (!is.na(future_vol[i])) {
          target_values[i] <- as.numeric(future_vol[i])
        }
      }
      
      features$target <- xts(target_values, order.by = index(features))
      
      return(validate_data(features, "ML Features"))
    }
    
    ml_data <- create_ml_features(returns)
    
    # Check sufficient data for ML analysis
    if (nrow(ml_data) < INITIAL_TRAIN_SIZE + TEST_LENGTH) {
      timestamp_message(paste("Insufficient data for ML backtest. Need at least", 
                              INITIAL_TRAIN_SIZE + TEST_LENGTH, "observations, have", nrow(ml_data)))
    } else {
      # --- 2.2: Rolling-Window ML Forecast Generation ---
      feature_cols <- grep("lag_vol|ma_vol", names(ml_data), value = TRUE)
      
      # Ensure we have complete cases for training
      complete_data <- na.omit(ml_data[, c("returns", feature_cols, "target")])
      
      if (nrow(complete_data) < INITIAL_TRAIN_SIZE) {
        timestamp_message("Insufficient complete cases for ML training")
      } else {
        n_periods <- floor((nrow(complete_data) - INITIAL_TRAIN_SIZE) / REFIT_EVERY)
        timestamp_message(paste("Starting ML rolling window backtest for", n_periods, "periods..."))
        
        # Sequential ML predictions (more robust than parallel for debugging)
        ml_predictions_list <- list()
        
        for (i in 1:n_periods) {
          train_end <- INITIAL_TRAIN_SIZE + (i - 1) * REFIT_EVERY
          test_start <- train_end + 1
          test_end <- min(test_start + REFIT_EVERY - 1, nrow(complete_data))
          
          if (test_start > nrow(complete_data)) break
          
          train_data <- complete_data[1:train_end, ]
          test_data <- complete_data[test_start:test_end, ]
          
          # Prepare training data
          train_features <- as.matrix(train_data[, feature_cols])
          train_target <- as.numeric(train_data$target)
          
          # Additional validation
          if (any(is.na(train_features)) || any(is.na(train_target)) || length(train_target) < 100) {
            timestamp_message(paste("Skipping period", i, "- insufficient clean data"))
            next
          }
          
          # Train XGBoost model
          xgb_model <- tryCatch({
            library(xgboost)
            dtrain <- xgb.DMatrix(data = train_features, label = train_target)
            
            xgb.train(
              params = list(
                objective = "reg:squarederror",
                eta = 0.1,
                max_depth = 4,
                subsample = 0.8,
                colsample_bytree = 0.8
              ),
              data = dtrain,
              nrounds = 100,
              verbose = 0
            )
          }, error = function(e) {
            timestamp_message(paste("XGBoost training failed in period", i, ":", e$message))
            NULL
          })
          
          if (!is.null(xgb_model)) {
            test_features <- as.matrix(test_data[, feature_cols])
            if (!any(is.na(test_features))) {
              dtest <- xgb.DMatrix(data = test_features)
              predictions <- predict(xgb_model, dtest)
              
              pred_xts <- xts(predictions, order.by = index(test_data))
              ml_predictions_list[[i]] <- pred_xts
              
              if (i %% 10 == 0) {
                timestamp_message(paste("Completed", i, "of", n_periods, "ML periods"))
              }
            }
          }
        }
        
        # Combine predictions
        if (length(ml_predictions_list) > 0) {
          # Remove NULL elements
          ml_predictions_list <- ml_predictions_list[!sapply(ml_predictions_list, is.null)]
          
          if (length(ml_predictions_list) > 0) {
            ml_predictions_xts <- do.call(rbind, ml_predictions_list)
            colnames(ml_predictions_xts) <- "ML_Pred"
            
            # --- 2.3: GARCH-X vs Standard GARCH Comparison ---
            # Align data robustly for comparison
            aligned_garch_data <- merge(returns, ml_predictions_xts, join = "inner")
            
            if (nrow(aligned_garch_data) > 100) {
              returns_for_garch <- aligned_garch_data[, "returns"]
              ml_regressor <- aligned_garch_data[, "ML_Pred"]
              
              spec_garch_x <- ugarchspec(
                variance.model = list(external.regressors = as.matrix(ml_regressor)),
                mean.model = list(armaOrder = c(1, 1)),
                distribution.model = "std"
              )
              
              spec_standard <- ugarchspec(
                variance.model = list(model = "sGARCH"),
                mean.model = list(armaOrder = c(1, 1)),
                distribution.model = "std"
              )
              
              fit_garch_x <- tryCatch({
                ugarchfit(spec_garch_x, data = returns_for_garch, solver = "hybrid")
              }, error = function(e) {
                timestamp_message(paste("GARCH-X fit failed:", e$message))
                NULL
              })
              
              fit_standard <- tryCatch({
                ugarchfit(spec_standard, data = returns_for_garch, solver = "hybrid")
              }, error = function(e) {
                timestamp_message(paste("Standard GARCH fit failed:", e$message))
                NULL
              })
              
              if (!is.null(fit_garch_x) && !is.null(fit_standard)) {
                cat("\n--- GARCH-X Information Criteria ---\n")
                print(infocriteria(fit_garch_x))
                cat("\n--- Standard GARCH Information Criteria ---\n")
                print(infocriteria(fit_standard))
                
                # Compare model performance
                aic_garch_x <- infocriteria(fit_garch_x)[1]
                aic_standard <- infocriteria(fit_standard)[1]
                
                cat("\n--- Model Comparison ---\n")
                cat(sprintf("GARCH-X AIC: %.4f\n", aic_garch_x))
                cat(sprintf("Standard GARCH AIC: %.4f\n", aic_standard))
                cat(sprintf("AIC Improvement: %.4f\n", aic_standard - aic_garch_x))
                
                save_checkpoint("section2_ml", c("ml_predictions_xts", "fit_garch_x", "fit_standard"))
                timestamp_message("ML-GARCH analysis completed successfully")
              }
            } else {
              timestamp_message("Insufficient aligned data for GARCH comparison")
            }
          } else {
            timestamp_message("No valid ML predictions generated")
          }
        } else {
          timestamp_message("No ML predictions generated")
        }
      }
    }
  }
} else {
  timestamp_message("XGBoost not available - running simplified volatility analysis")
  
  # Simplified analysis without ML
  tryCatch({
    getSymbols("^GSPC", src = "yahoo", from = "2015-01-01", to = Sys.Date())
    returns <- dailyReturn(GSPC, type = "log") * 100
    returns <- returns[-1, ]
    colnames(returns) <- "returns"
    
    # Simple GARCH model
    spec_simple <- ugarchspec(variance.model = list(model = "sGARCH"), 
                              mean.model = list(armaOrder = c(1, 1)))
    fit_simple <- ugarchfit(spec_simple, data = returns)
    
    if (!is.null(fit_simple)) {
      cat("\n--- Simple GARCH Model Results ---\n")
      print(infocriteria(fit_simple))
      save_checkpoint("section2_simple", c("fit_simple", "returns"))
    }
  }, error = function(e) {
    timestamp_message(paste("Simple GARCH analysis failed:", e$message))
  })
}

monitor_resources()
timestamp_message("Section 2 Complete.")


# #############################################################################
# SECTION 3: HIGH-FREQUENCY DATA INTEGRATION
# #############################################################################
timestamp_message("Starting Section 3: High-Frequency Analysis...")

# --- 3.1: Robust Realized Volatility Data Acquisition ---
acquire_realized_volatility <- function(fallback_returns) {
  rv_file <- "realized_vol_cache.rds"
  
  if (file.exists(rv_file)) {
    timestamp_message("Loading realized volatility from cache")
    return(readRDS(rv_file))
  }
  
  vix_data <- tryCatch({
    timestamp_message("Attempting to load VIX data from FRED")
    getSymbols("VIXCLS", src = "FRED", from = "2010-01-01", auto.assign = FALSE)
  }, error = function(e) {
    timestamp_message(paste("FRED VIX failed:", e$message)); NULL
  })
  
  if (!is.null(vix_data)) {
    vix_data <- na.omit(vix_data)
    colnames(vix_data) <- "VIX"
    saveRDS(vix_data, file = rv_file)
    timestamp_message("VIX data successfully acquired and cached")
    return(vix_data)
  }
  
  timestamp_message("Generating realistic simulated realized volatility as fallback")
  actual_vol <- rollapply(fallback_returns, 20, sd, align = "right") * sqrt(252)
  actual_vol <- na.omit(actual_vol)
  sim_rv <- as.numeric(actual_vol) + rnorm(length(actual_vol), 0, 2)
  sim_rv <- pmax(sim_rv, 5)
  rv_sim <- xts(sim_rv, order.by = index(actual_vol))
  saveRDS(rv_sim, file = rv_file)
  return(rv_sim)
}

# Ensure we have returns data
if (!exists("returns") || is.null(returns)) {
  tryCatch({
    getSymbols("^GSPC", src = "yahoo", from = "2010-01-01", to = Sys.Date())
    returns <- dailyReturn(GSPC, type = "log") * 100
    returns <- returns[-1, ]; colnames(returns) <- "returns"
  }, error = function(e) {
    timestamp_message("Cannot proceed with Section 3 - no return data available")
    returns <- NULL
  })
}

if (!is.null(returns)) {
  rv_data <- acquire_realized_volatility(returns)
  colnames(rv_data) <- "RV"
  rv_data <- validate_data(rv_data, "Realized Volatility Data")
  
  # --- 3.2: HAR-RV Model Implementation ---
  prepare_har_data <- function(rvs) {
    d <- rvs
    d$weekly <- rollmean(rvs, 5, align = "right", fill = NA)
    d$monthly <- rollmean(rvs, 22, align = "right", fill = NA)
    d$daily_lag <- lag(d[, 1], 1)
    d$weekly_lag <- lag(d$weekly, 1)
    d$monthly_lag <- lag(d$monthly, 1)
    return(na.omit(d))
  }
  
  har_data <- prepare_har_data(rv_data)
  
  if (nrow(har_data) > 100) {
    har_model <- tryCatch({
      lm(RV ~ daily_lag + weekly_lag + monthly_lag, data = har_data)
    }, error = function(e) {
      timestamp_message(paste("HAR model fit failed:", e$message)); NULL
    })
    
    if (!is.null(har_model)) {
      # --- FIXED DATA ALIGNMENT FOR GARCH-RV ---
      rv_regressor <- lag(rv_data$RV, 1)
      colnames(rv_regressor) <- "RV_Lagged"
      aligned_data <- merge(returns, rv_regressor, join = "inner")
      aligned_data <- na.omit(aligned_data)
      
      if (nrow(aligned_data) > 500) {
        returns_hf <- aligned_data[, "returns"]
        regressor_hf <- aligned_data[, "RV_Lagged"]
        
        spec_garch_rv <- ugarchspec(
          variance.model = list(external.regressors = as.matrix(regressor_hf)),
          mean.model = list(armaOrder = c(1, 1)),
          distribution.model = "std"
        )
        
        fit_garch_rv <- tryCatch({
          ugarchfit(spec_garch_rv, data = returns_hf, solver = "hybrid")
        }, error = function(e) {
          timestamp_message(paste("GARCH-RV fit failed:", e$message)); NULL
        })
        
        if (!is.null(fit_garch_rv)) {
          cat("\n--- HAR Model Summary ---\n")
          print(summary(har_model))
          cat("\n--- GARCH with Realized Volatility Results ---\n")
          print(infocriteria(fit_garch_rv))
          
          save_checkpoint("section3_hf", c("rv_data", "har_model", "fit_garch_rv"))
          timestamp_message("High-frequency analysis completed successfully")
        }
      } else {
        timestamp_message("Insufficient aligned data for GARCH-RV analysis")
      }
    }
  } else {
    timestamp_message("Insufficient data for HAR analysis")
  }
}

monitor_resources()
timestamp_message("Section 3 Complete.")


# #############################################################################
# SECTION 4: RISK MANAGEMENT BACKTEST
# #############################################################################
timestamp_message("Starting Section 4: VaR Backtest...")

if (!exists("returns") || is.null(returns)) {
  tryCatch({
    getSymbols("^GSPC", src = "yahoo", from = "2015-01-01", to = Sys.Date())
    returns <- dailyReturn(GSPC, type = "log") * 100
    returns <- returns[-1, ]; colnames(returns) <- "returns"
  }, error = function(e) {
    timestamp_message("Cannot get data for VaR backtest"); returns <- NULL
  })
}

if (!is.null(returns) && nrow(returns) >= BACKTEST_WINDOW + TEST_LENGTH) {
  # --- 4.1: Rolling VaR Backtest ---
  available_data <- nrow(returns) - BACKTEST_WINDOW
  n_roll_periods <- min(floor(available_data / REFIT_EVERY), floor(TEST_LENGTH / REFIT_EVERY))
  
  timestamp_message(paste("Starting VaR backtest for", n_roll_periods, "rolling periods..."))
  
  model_specs <- list(
    Standard = ugarchspec(variance.model = list(model = "sGARCH"), 
                          mean.model = list(armaOrder = c(1, 1)), 
                          distribution.model = "std"),
    GARCH11 = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), 
                         mean.model = list(armaOrder = c(0, 0)), 
                         distribution.model = "norm"),
    eGARCH_Asymmetric = ugarchspec(variance.model = list(model = "eGARCH"),
                                   mean.model = list(armaOrder = c(1,1)),
                                   distribution.model = "sstd")
  )
  
  if (HAS_RMGARCH) {
    model_specs$CGARCH <- ugarchspec(variance.model = list(model = "csGARCH"), 
                                     mean.model = list(armaOrder = c(1, 1)), 
                                     distribution.model = "std")
  }
  
  backtest_results <- data.frame()
  
  for (t in 1:n_roll_periods) {
    timestamp_message(paste("Processing rolling period", t, "of", n_roll_periods))
    
    # --- FIXED WINDOWING LOGIC TO PREVENT INDEX ERRORS ---
    train_start_idx <- (t - 1) * REFIT_EVERY + 1
    train_end_idx <- train_start_idx + BACKTEST_WINDOW - 1
    
    test_start_idx <- train_end_idx + 1
    test_end_idx <- min(test_start_idx + REFIT_EVERY - 1, nrow(returns))
    
    # Safety check to ensure we don't run past the end of the data
    if (train_end_idx > nrow(returns) || test_start_idx > nrow(returns) || test_end_idx < test_start_idx) {
      timestamp_message(paste("Stopping at period", t, "- insufficient remaining data"))
      break
    }
    
    train_data <- returns[train_start_idx:train_end_idx, ]
    test_data <- returns[test_start_idx:test_end_idx, ]
    
    # Validate training data
    if (nrow(train_data) < 100 || nrow(test_data) < 1) {
      timestamp_message(paste("Skipping period", t, "- insufficient data"))
      next
    }
    
    for (model_name in names(model_specs)) {
      tryCatch({
        fit <- ugarchfit(model_specs[[model_name]], data = train_data, solver = 'hybrid')
        
        if (convergence(fit) == 0) {
          forecasts <- ugarchforecast(fit, n.ahead = nrow(test_data))
          sigma_f <- as.numeric(sigma(forecasts))
          mu_f <- as.numeric(fitted(forecasts))
          
          # Handle single vs multiple forecasts
          if (length(sigma_f) == 1) {
            sigma_f <- rep(sigma_f, nrow(test_data))
          }
          if (length(mu_f) == 1) {
            mu_f <- rep(mu_f, nrow(test_data))
          }
          
          var_f <- qnorm(VAR_ALPHA, mean = mu_f, sd = sigma_f)
          
          period_results <- data.frame(
            Model = model_name, 
            VaR = var_f, 
            Actual = as.numeric(test_data), 
            Period = t, 
            Date = index(test_data),
            stringsAsFactors = FALSE
          )
          backtest_results <- rbind(backtest_results, period_results)
        } else {
          timestamp_message(paste("Model", model_name, "failed to converge in period", t))
        }
      }, error = function(e) {
        timestamp_message(paste("Model", model_name, "failed in period", t, ":", e$message))
      })
    }
    
    # Progress update every 5 periods
    if (t %% 5 == 0) {
      monitor_resources()
    }
  }
  
  # --- 4.2: Analyze VaR Backtest Results ---
  if (nrow(backtest_results) > 0) {
    cat("\n=== VaR BACKTEST RESULTS ===\n")
    
    for (model_name in unique(backtest_results$Model)) {
      cat(paste("\n--- VaR Results for:", model_name, "---\n"))
      subset_data <- backtest_results[backtest_results$Model == model_name, ]
      
      if (nrow(subset_data) > 10) {
        violations <- sum(subset_data$Actual < subset_data$VaR, na.rm = TRUE)
        violation_rate <- violations / nrow(subset_data)
        expected_rate <- VAR_ALPHA
        
        cat(sprintf("Observations: %d\n", nrow(subset_data)))
        cat(sprintf("Violations: %d\n", violations))
        cat(sprintf("Violation Rate: %.3f%% (Expected: %.3f%%)\n", 
                    violation_rate * 100, expected_rate * 100))
        
        # FIXED: Statistical test for VaR accuracy using the correct package
        if (requireNamespace("rugarch", quietly = TRUE)) {
          var_test_result <- tryCatch({
            rugarch::VaRTest(alpha = VAR_ALPHA, 
                             actual = subset_data$Actual, 
                             VaR = subset_data$VaR)
          }, error = function(e) {
            cat("VaR test failed:", e$message, "\n")
            NULL
          })
          
          if (!is.null(var_test_result)) {
            print(var_test_result)
          }
        }
        
        # Additional diagnostics
        avg_var <- mean(subset_data$VaR, na.rm = TRUE)
        avg_return <- mean(subset_data$Actual, na.rm = TRUE)
        cat(sprintf("Average VaR: %.4f\n", avg_var))
        cat(sprintf("Average Return: %.4f\n", avg_return))
        
        # Calculate coverage ratio
        coverage_ratio <- violation_rate / expected_rate
        cat(sprintf("Coverage Ratio: %.3f ", coverage_ratio))
        if (coverage_ratio > 1.5) {
          cat("(Model may be too conservative)\n")
        } else if (coverage_ratio < 0.5) {
          cat("(Model may be too aggressive)\n")
        } else {
          cat("(Good coverage)\n")
        }
      }
    }
    
    # --- 4.3: Visualize VaR Performance ---
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      # Create summary plot
      library(ggplot2)
      
      # Calculate violation rates by model
      violation_summary <- aggregate(Actual < VaR ~ Model, data = backtest_results, FUN = mean, na.rm = TRUE)
      names(violation_summary)[2] <- "Violation_Rate"
      
      plot_data <- reshape2::melt(violation_summary, id.vars = "Model", 
                                  variable.name = "Type", value.name = "Rate")
      
      p <- ggplot(plot_data, aes(x = Model, y = Rate * 100, fill = Model)) +
        geom_bar(stat = "identity", position = "dodge") +
        geom_hline(yintercept = VAR_ALPHA * 100, linetype = "dashed", color = "red") +
        labs(title = "VaR Model Performance Comparison",
             subtitle = paste("Expected violation rate:", VAR_ALPHA * 100, "% (Red Dashed Line)"),
             y = "Violation Rate (%)", x = "Model") +
        theme_minimal() +
        guides(fill = "none") # Remove the legend as model names are on the x-axis
      
      print(p)
    }
    
    save_checkpoint("section4_var", c("backtest_results"))
    timestamp_message("VaR backtest analysis completed successfully")
  } else {
    timestamp_message("No VaR backtest results generated")
  }
} else {
  timestamp_message(paste("Insufficient data for VaR backtest. Need", BACKTEST_WINDOW + TEST_LENGTH, 
                          "obs, have", ifelse(exists("returns"), nrow(returns), 0)))
}

monitor_resources()
timestamp_message("Section 4 Complete.")


# #############################################################################
# SECTION 5: COMPREHENSIVE RESULTS SUMMARY
# #############################################################################
timestamp_message("Starting Section 5: Results Summary...")

# --- 5.1: Generate Executive Summary ---
generate_summary <- function() {
  # FIXED: Replaced '+' for string concatenation with cat() for printing
  cat("\n")
  cat(paste(rep("=", 80), collapse = ""))
  cat("\n                    VOLATILITY MODELING ANALYSIS SUMMARY\n")
  cat(paste(rep("=", 80), collapse = ""))
  cat("\n\n")
  
  cat("Analysis Date:", format(Sys.Date(), "%Y-%m-%d"), "\n")
  cat("Execution Time:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  
  # Check what analyses were completed
  completed_sections <- character()
  
  if (file.exists("checkpoint_section1_dcc.RData") || file.exists("checkpoint_section1_simple.RData")) {
    completed_sections <- c(completed_sections, "Portfolio Correlation Analysis")
  }
  
  if (file.exists("checkpoint_section2_ml.RData") || file.exists("checkpoint_section2_simple.RData")) {
    completed_sections <- c(completed_sections, "ML-GARCH Integration")
  }
  
  if (file.exists("checkpoint_section3_hf.RData")) {
    completed_sections <- c(completed_sections, "High-Frequency Data Analysis")
  }
  
  if (file.exists("checkpoint_section4_var.RData")) {
    completed_sections <- c(completed_sections, "VaR Backtesting")
  }
  
  cat("COMPLETED ANALYSES:\n")
  if (length(completed_sections) > 0) {
    for (i in seq_along(completed_sections)) {
      cat(paste(i, ".", completed_sections[i], "\n"))
    }
  } else {
    cat("No analyses completed successfully.\n")
  }
  
  cat("\nSYSTEM CAPABILITIES DETECTED:\n")
  cat("- Multivariate GARCH:", ifelse(HAS_RMGARCH, "AVAILABLE", "NOT AVAILABLE"), "\n")
  cat("- Machine Learning:", ifelse(HAS_XGBOOST, "AVAILABLE", "NOT AVAILABLE"), "\n")
  cat("- Parallel Processing:", ifelse(n_cores > 1, paste("ENABLED (", n_cores, " cores)"), "DISABLED"), "\n")
  
  # Final resource check
  final_mem <- gc()[2, 2]
  cat("\nFINAL MEMORY USAGE:", round(final_mem, 1), "MB\n")
  
  cat("\nCHECKPOINT FILES CREATED:\n")
  checkpoint_files <- list.files(pattern = "^checkpoint_.*\\.RData$")
  if (length(checkpoint_files) > 0) {
    for (file in checkpoint_files) {
      cat("-", file, "\n")
    }
  } else {
    cat("No checkpoint files created.\n")
  }
  
  cat("\n")
  cat(paste(rep("=", 80), collapse = ""))
  cat("\nAnalysis framework execution completed.\n")
  cat("Use load() function to restore any checkpoint for further analysis.\n")
  cat(paste(rep("=", 80), collapse = ""))
  cat("\n")
}

generate_summary()

# --- 5.2: Optional: Quick Model Comparison if Multiple Models Fitted ---
# This section provides a consolidated comparison of models fitted across the script.
tryCatch({
  models_list <- list()
  
  if (exists("fit_standard")) {
    models_list[["Std_GARCH_from_ML"]] <- fit_standard
  }
  if (exists("fit_garch_x")) {
    models_list[["GARCH-X_from_ML"]] <- fit_garch_x
  }
  if (exists("fit_garch_rv")) {
    models_list[["GARCH-RV_from_HF"]] <- fit_garch_rv
  }
  
  if (length(models_list) > 1) {
    cat("\n=== COMPREHENSIVE MODEL COMPARISON ===\n")
    
    comparison_table <- data.frame(
      Model = names(models_list),
      AIC = sapply(models_list, function(m) infocriteria(m)[1]),
      BIC = sapply(models_list, function(m) infocriteria(m)[2]),
      Log_Likelihood = sapply(models_list, function(m) likelihood(m)),
      stringsAsFactors = FALSE
    )
    
    print(comparison_table, row.names = FALSE)
    
    best_aic <- comparison_table$Model[which.min(comparison_table$AIC)]
    cat(paste("\nBest model by AIC:", best_aic, "\n"))
  }
}, error = function(e){
  timestamp_message(paste("Could not generate final model comparison:", e$message))
})


# --- 5.3: Cleanup and Final Steps ---
# Clean up temporary files
temp_files <- c("write_test.tmp")
for (file in temp_files) {
  if (file.exists(file)) {
    unlink(file)
  }
}
# Note: "realized_vol_cache.rds" is kept for faster re-runs.

# Stop parallel cluster if running
if (requireNamespace("doParallel", quietly = TRUE) && n_cores > 1) {
  tryCatch({
    stopImplicitCluster()
    timestamp_message("Parallel cluster stopped")
  }, error = function(e) {
    timestamp_message(paste("Error stopping cluster:", e$message))
  })
}

timestamp_message("Research framework execution finished successfully.")
