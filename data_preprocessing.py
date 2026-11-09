import numpy as np
import pandas as pd


# 1. Load and clean data

df = pd.read_csv("used_cars.csv")
original_rows = len(df)

df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={"milage": "mileage"})

missing_values = [
    "", " ", "-", "–", "—", "−",
    "N/A", "n/a", "None", "none",
    "Unknown", "unknown", "null", "NULL"
]

for column in df.select_dtypes(include="object"):
    df[column] = df[column].astype("string").str.strip()
    df[column] = df[column].replace(missing_values, pd.NA)

df["price"] = pd.to_numeric(
    df["price"].astype("string")
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

df["mileage"] = pd.to_numeric(
    df["mileage"].astype("string")
    .str.replace(",", "", regex=False)
    .str.replace("miles", "", case=False, regex=False)
    .str.replace("mile", "", case=False, regex=False)
    .str.replace("mi", "", case=False, regex=False)
    .str.strip().str.rstrip(".").str.strip(),
    errors="coerce"
)

df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")


# 2. Standardize fuel categories

fuel_mapping = {
    "gas": "Gasoline",
    "gasoline": "Gasoline",
    "diesel": "Diesel",
    "hybrid": "Hybrid",
    "hybrid electric": "Hybrid",
    "gas/electric": "Hybrid",
    "mild hybrid": "Hybrid",
    "mhev": "Hybrid",
    "hev": "Hybrid",
    "plug-in hybrid": "Plug-In Hybrid",
    "plug in hybrid": "Plug-In Hybrid",
    "phev": "Plug-In Hybrid",
    "e85 flex fuel": "E85 Flex Fuel",
    "electric": "Electric",
    "battery electric": "Electric",
    "battery electric vehicle": "Electric",
    "bev": "Electric"
}

df["fuel_type"] = (
    df["fuel_type"].astype("string").str.strip().str.lower()
    .replace(fuel_mapping)
)


# 3. Extract engine information

def get_engine_info(engine):

    if pd.isna(engine):
        return np.nan, np.nan, np.nan, "Missing"

    words = (
        str(engine).lower()
        .replace("-", " ")
        .replace(",", "")
        .replace("/", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )

    horsepower = np.nan
    liters = np.nan
    cylinders = np.nan
    engine_type = "Other"

    # Horsepower

    i = 0

    while i < len(words):
        word = words[i]

        if word.endswith("horsepower"):
            value = word.removesuffix("horsepower")
        elif word.endswith("hp"):
            value = word.removesuffix("hp")
        elif (
            word.replace(".", "", 1).isdigit()
            and i + 1 < len(words)
            and words[i + 1] in ["hp", "horsepower"]
        ):
            value = word
        else:
            i += 1
            continue

        try:
            horsepower = float(value)
            break
        except ValueError:
            i += 1

    # Engine displacement

    i = 0

    while i < len(words):
        word = words[i]

        try:
            if word.endswith("cc"):
                liters = float(word.removesuffix("cc")) / 1000
                break

            if word.endswith("l"):
                liters = float(word.removesuffix("l"))
                break

            if (
                word.replace(".", "", 1).isdigit()
                and i + 1 < len(words)
                and words[i + 1] in ["l", "liter", "liters", "cc"]
            ):
                liters = float(word)

                if words[i + 1] == "cc":
                    liters /= 1000

                break

        except ValueError:
            pass

        i += 1

    # Cylinder count and configuration

    configurations = {
        "v": "V",
        "i": "Inline",
        "h": "Flat",
        "w": "W"
    }

    i = 0

    while i < len(words):
        word = words[i]

        if word and word[0] in configurations:
            digits = word[1:3] if word[1:3].isdigit() else word[1:2]

            if digits.isdigit() and 1 <= int(digits) <= 16:
                cylinders = int(digits)
                engine_type = configurations[word[0]]
                break

        if word in configurations and i + 1 < len(words):
            if words[i + 1].isdigit() and 1 <= int(words[i + 1]) <= 16:
                cylinders = int(words[i + 1])
                engine_type = configurations[word]
                break

        if word in ["inline", "straight", "flat", "boxer"]:
            engine_type = "Flat" if word in ["flat", "boxer"] else "Inline"

            if i + 1 < len(words) and words[i + 1].isdigit():
                if 1 <= int(words[i + 1]) <= 16:
                    cylinders = int(words[i + 1])

            break

        if word.isdigit() and 1 <= int(word) <= 16 and i + 1 < len(words):
            if words[i + 1].rstrip("s") in ["cylinder", "cyl"]:
                cylinders = int(word)
                break

        value = (
            word.removesuffix("cylinders")
            .removesuffix("cylinder")
            .removesuffix("cyl")
        )

        if value.isdigit() and 1 <= int(value) <= 16:
            cylinders = int(value)
            break

        i += 1

    if any(word in words for word in ["rotary", "rotor", "wankel"]):
        engine_type = "Rotary"

    return horsepower, liters, cylinders, engine_type


engine_columns = [
    "horsepower", "engine_liters", "cylinders", "engine_type"
]

df[engine_columns] = pd.DataFrame(
    df["engine"].apply(get_engine_info).tolist(),
    index=df.index,
    columns=engine_columns
)


# 4. Identify fully electric vehicles

def is_full_electric(row):

    fuel = str(row["fuel_type"]).lower()
    engine = str(row["engine"]).lower()

    words = (
        engine.replace("-", " ")
        .replace(",", "")
        .replace("/", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )

    if any(word in words for word in ["hybrid", "phev", "mhev", "hev"]):
        return False

    if fuel in ["hybrid", "plug-in hybrid"]:
        return False

    if fuel in ["electric", "battery electric", "bev"]:
        return True

    return any(word in words for word in ["electric", "battery", "bev"])


electric = df.apply(is_full_electric, axis=1)
non_electric = ~electric

df["is_electric"] = electric.astype(int)
df.loc[electric, "fuel_type"] = "Electric"
df.loc[electric, "engine_type"] = "Electric"
df.loc[electric, ["engine_liters", "cylinders"]] = 0


# 5. Engine extraction audit

print("\nENGINE EXTRACTION AUDIT")
print("------------------------------")
print("Fully electric vehicles:", int(electric.sum()))
print("Non-electric vehicles:", int(non_electric.sum()))

for column in ["horsepower", "engine_liters", "cylinders"]:
    missing = df.loc[non_electric, column].isna().sum()
    percentage = missing / non_electric.sum() * 100
    print(f"{column} missing: {missing} ({percentage:.2f}%)")

df = df.drop(columns="engine")


# 6. Standardize categories and transmission

for column in [
    "brand", "model", "fuel_type",
    "transmission", "ext_col", "int_col"
]:
    df[column] = df[column].astype("string").str.strip().str.title()

df["transmission"] = (
    df["transmission"]
    .str.replace("Dualshift", "Dual Shift", case=False, regex=False)
    .str.replace("W/", "w/", regex=False)
)


# 7. Convert accident and clean-title values

def convert_accident(value):

    if pd.isna(value):
        return np.nan

    value = str(value).lower()

    if value in ["no", "false", "0"] or "none" in value:
        return 0

    if value in ["yes", "true", "1"] or "accident" in value or "damage" in value:
        return 1

    return np.nan


def convert_clean_title(value):

    if pd.isna(value):
        return np.nan

    value = str(value).lower()

    if value in ["yes", "true", "1"]:
        return 1

    if value in ["no", "false", "0"]:
        return 0

    return np.nan


df["accident"] = df["accident"].apply(convert_accident)
df["clean_title"] = df["clean_title"].apply(convert_clean_title)


# 8. Remove invalid records

df = df.dropna(subset=["price"])
df = df[df["price"] > 0]
df = df[df["mileage"].isna() | (df["mileage"] >= 0)]
df = df.drop_duplicates()

df.loc[df["engine_liters"] > 15, "engine_liters"] = np.nan
df.loc[df["cylinders"] > 16, "cylinders"] = np.nan
df.loc[df["horsepower"] > 2000, "horsepower"] = np.nan


# 9. Detect outliers using IQR

measurements = [
    "price", "mileage", "horsepower", "engine_liters", "cylinders"
]

for column in measurements:

    values = (
        df.loc[non_electric, column].dropna()
        if column in ["engine_liters", "cylinders"]
        else df[column].dropna()
    )

    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1

    df[column + "_outlier"] = (
        (df[column] < q1 - 1.5 * iqr)
        | (df[column] > q3 + 1.5 * iqr)
    ).fillna(False)

    df[column + "_status"] = np.select(
        [df[column].isna(), df[column + "_outlier"]],
        ["Missing", "Outlier"],
        default="Normal"
    )


# 10. Classify outliers and identify vehicles

outlier_flags = [column + "_outlier" for column in measurements]

df["outlier_count"] = df[outlier_flags].sum(axis=1)

df["outlier_category"] = np.where(
    df["outlier_count"] > 0, "Outlier", "Not Outlier"
)

df["outlier_types"] = ""

for column in measurements:
    df["outlier_types"] += np.where(
        df[column + "_outlier"], column + ", ", ""
    )

df["outlier_types"] = df["outlier_types"].str.rstrip(", ")
df.loc[df["outlier_types"] == "", "outlier_types"] = "None"

year = df["model_year"].astype("Int64").astype("string").fillna("Missing")

df["vehicle"] = (
    year + " "
    + df["brand"].fillna("Missing") + " "
    + df["model"].fillna("Missing")
)


# 11. Arrange and save final dataset

base_columns = [
    "vehicle", "brand", "model", "model_year", "mileage", "price",
    "fuel_type", "is_electric", "horsepower", "engine_liters",
    "cylinders", "engine_type", "transmission", "ext_col",
    "int_col", "accident", "clean_title"
]

status_columns = [
    column + suffix
    for column in measurements
    for suffix in ["_outlier", "_status"]
]

df = df[base_columns + status_columns + [
    "outlier_count", "outlier_category", "outlier_types"
]]

df.to_csv("used_cars_clean.csv", index=False)


# 12. Display results

print("\nPREPROCESSING RESULTS")
print("------------------------------")
print("Original rows:", original_rows)
print("Cleaned rows:", len(df))
print("Removed rows:", original_rows - len(df))

print("\nOUTLIER SUMMARY")
print("------------------------------")
print(df["outlier_category"].value_counts())

print("\nOUTLIER COUNTS BY MEASUREMENT")
print("------------------------------")

for column in measurements:
    print(column, ":", int(df[column + "_outlier"].sum()))

print("\nMISSING VALUES")
print("------------------------------")

missing = df.isna().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percentage": (missing / len(df) * 100).round(2).values
    }).to_string(index=False))

print("\nTOP 50 VEHICLES WITH OUTLIERS OR MISSING MEASUREMENTS")
print("------------------------------")

report = df[
    df["outlier_category"].eq("Outlier")
    | df[measurements].isna().any(axis=1)
]

print(report.sort_values(
    ["outlier_count", "vehicle"],
    ascending=[False, True]
)[[
    "vehicle", "outlier_category", "outlier_types", "outlier_count"
]].head(50).to_string(index=False))

print("\nENGINE TYPE COUNTS")
print("------------------------------")
print(df["engine_type"].value_counts(dropna=False))

print("\nFUEL TYPE COUNTS")
print("------------------------------")
print(df["fuel_type"].value_counts(dropna=False))

print("\nELECTRIC VEHICLE COUNTS")
print("------------------------------")
print(df["is_electric"].value_counts())

print("\nMODEL YEAR RANGE")
print("------------------------------")
print("Minimum year:", df["model_year"].min())
print("Maximum year:", df["model_year"].max())

print("\nOUTPUT FILE")
print("------------------------------")
print("used_cars_clean.csv")