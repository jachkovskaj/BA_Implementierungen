library(readxl)
library(SPlit)
library(writexl)

read_text <- function(path) {
  if (!file.exists(path)) stop(paste("Missing file:", path))
  readLines(path, warn = FALSE)
}

name <- read_text("temp_filename.txt")[1]
val_ratio <- as.numeric(read_text("val_ratio.txt"))
seed_value <- as.numeric(read_text("seed.txt"))

if (is.na(val_ratio) || val_ratio <= 0 || val_ratio >= 1) {
  stop("val_ratio must be between 0 and 1")
}
if (is.na(seed_value)) stop("seed invalid")

file_name <- paste0(name, ".xlsx")
data <- read_excel(file_name)

# tibble -> data.frame (SPlit ist hier oft picky)
data <- as.data.frame(data)
print(sapply(data, class))

set.seed(seed_value)

# Wenn letzte Spalte character ist, entfernen (wie du wolltest)
if (is.character(data[[ncol(data)]])) {
  data <- data[, -ncol(data), drop = FALSE]
}

# Alle Spalten in SPlit-kompatible Typen überführen:
for (j in seq_along(data)) {
  x <- data[[j]]

  # logical -> numeric
  if (is.logical(x)) {
    data[[j]] <- as.numeric(x)
    next
  }

  # Date/POSIXct -> numeric (Zeitstempel) oder entferne, wenn du sie nicht willst
  if (inherits(x, "Date") || inherits(x, "POSIXct") || inherits(x, "POSIXt")) {
    data[[j]] <- as.numeric(x)
    next
  }

  # character -> versuche numeric zu parsen (auch Komma-Dezimal)
  if (is.character(x)) {
    x2 <- gsub(",", ".", x)            # "1,23" -> "1.23"
    xn <- suppressWarnings(as.numeric(x2))

    # Wenn viele NAs durch die Umwandlung entstehen -> als factor statt numeric
    # (SPlit erlaubt factor)
    if (sum(is.na(xn)) > sum(is.na(x)) ) {
      data[[j]] <- as.factor(x)
    } else {
      data[[j]] <- xn
    }
    next
  }

  # sonst: numeric/integer/factor OK
}

# finale Sicherheit: nur numeric oder factor behalten
ok_cols <- sapply(data, function(x) is.numeric(x) || is.factor(x))
if (!all(ok_cols)) {
  bad <- names(data)[!ok_cols]
  stop(paste("Non-numeric/non-factor columns remain:", paste(bad, collapse = ", ")))
}

combined_data <- data

val_idx <- SPlit(combined_data, splitRatio = val_ratio)
dataVal <- combined_data[val_idx, , drop = FALSE]
dataTrain <- combined_data[-val_idx, , drop = FALSE]

write_xlsx(dataTrain, path = paste0(name, "_train.xlsx"))
write_xlsx(dataVal,   path = paste0(name, "_val.xlsx"))