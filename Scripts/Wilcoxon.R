# Load the CSV file
data <- read.csv("MIscorepvalue.csv")

# Convert columns to numeric (if necessary)
data$MI.score.family <- as.numeric(as.character(data$MI.score.family))
data$MI.Score.GO <- as.numeric(as.character(data$MI.Score.GO))

# Perform the Wilcoxon test between MI.score.family and MI.Score.GO columns
wilcox_test <- wilcox.test(data$MI.score.family, data$MI.Score.GO, paired = TRUE, na.rm = TRUE)

# Print the test result
print(wilcox_test)

