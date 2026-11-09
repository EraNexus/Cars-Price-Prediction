Issues with data_preprocessing

The dataset has missing values and inconsistencies that can affect model performance.

Missing horsepower: 808 records (20.15%). Some engine descriptions lack horsepower information, and reliable estimates cannot be determined.

Missing engine information: 56 records lack engine displacement, and 216 lack cylinder counts.

Missing vehicle information: 49 fuel types, 4 transmissions, 15 exterior colors, and 133 interior colors are missing.

Missing vehicle history: 113 accident-history values and 596 clean-title values are missing.

Incomplete engine descriptions: Some descriptions do not provide enough information to identify engine specifications. Unidentified configurations are classified as Other.

Outliers: 363 vehicles have at least one statistical outlier in price, mileage, or engine measurements. These are flagged using the IQR method but are not removed because they may represent legitimate vehicles.

Handling: Missing values are preserved rather than replaced with potentially inaccurate estimates. Invalid prices, negative mileage, and exact duplicate records are removed. Remaining data issues will be evaluated during feature engineering and model development.