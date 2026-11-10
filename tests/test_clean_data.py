import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import src.clean_data as clean_data
from src.clean_data import (
    assign_age_group,
    assign_fare_quartile,
    cleaner,
    engineer_cabin_features,
    get_input_filename,
    impute_age,
    impute_embarked,
    load_data,
    plot_age_distribution,
    plot_class_survival,
    plot_correlation_heatmap,
    plot_fare_distribution,
    plot_gender_survival,
    plot_multivariate_class_sex,
    plot_overall_survival,
    print_menu,
    save_cleaned_data,
    validate_columns,
    visualization_menu,
)


PLOT_FUNCTIONS = [
    plot_overall_survival,
    plot_gender_survival,
    plot_class_survival,
    plot_age_distribution,
    plot_fare_distribution,
    plot_multivariate_class_sex,
    plot_correlation_heatmap,
]

MENU_DISPATCH = {
    "1": "plot_overall_survival",
    "2": "plot_gender_survival",
    "3": "plot_class_survival",
    "4": "plot_age_distribution",
    "5": "plot_fare_distribution",
    "6": "plot_multivariate_class_sex",
    "7": "plot_correlation_heatmap",
}


# Fixtures

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Synthetic Titanic-schema DataFrame with edge cases.

    Rows 5 and 6 are exact duplicates (PassengerId 6).
    Embarked mode is 'S'. Cabin is missing for most rows.
    """
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6, 6],
        "Survived": [0, 1, 1, 0, 1, 0, 0],
        "Pclass": [1, 1, 2, 2, 3, 3, 3],
        "Name": ["Braund", "Cumings", "Heikkinen", "Futrelle", "Allen", "Moran", "Moran"],
        "Sex": ["male", "female", "female", "male", "female", "male", "male"],
        "Age": [22.0, 38.0, 26.0, np.nan, np.nan, 25.0, 25.0],
        "SibSp": [1, 1, 0, 0, 0, 0, 0],
        "Parch": [0, 0, 0, 0, 2, 0, 0],
        "Ticket": ["A/5", "PC", "STON", "113", "347", "370", "370"],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 10.0, 10.0],
        "Cabin": [np.nan, "C85", np.nan, "C123", np.nan, "E46", "E46"],
        "Embarked": ["S", "C", "S", "S", np.nan, "Q", "Q"],
    })


@pytest.fixture
def sample_clean_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned version of the sample data, used by visualization and menu tests."""
    return cleaner(sample_raw_df)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to prevent memory build-up."""
    yield
    plt.close("all")


# 1. CLI Argument Tests

def test_get_input_filename_default() -> None:
    assert get_input_filename([]) == "train.csv"


def test_get_input_filename_short_flag() -> None:
    assert get_input_filename(["-f", "custom.csv"]) == "custom.csv"


def test_get_input_filename_long_flag() -> None:
    assert get_input_filename(["--file", "test_data.csv"]) == "test_data.csv"


def test_get_input_filename_unknown_flag_exits() -> None:
    """argparse exits with SystemExit on unrecognized arguments."""
    with pytest.raises(SystemExit):
        get_input_filename(["--unknown"])


# 2. File I/O Tests

def test_load_data_success(tmp_path: Path, sample_raw_df: pd.DataFrame) -> None:
    csv_file = tmp_path / "valid.csv"
    sample_raw_df.to_csv(csv_file, index=False)

    loaded_df = load_data(csv_file)
    assert loaded_df.shape == sample_raw_df.shape
    assert list(loaded_df.columns) == list(sample_raw_df.columns)


def test_load_data_accepts_string_path(tmp_path: Path, sample_raw_df: pd.DataFrame) -> None:
    csv_file = tmp_path / "valid.csv"
    sample_raw_df.to_csv(csv_file, index=False)

    loaded_df = load_data(str(csv_file))
    assert not loaded_df.empty


def test_load_data_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="was not found"):
        load_data(tmp_path / "non_existent.csv")


def test_load_data_invalid_extension(tmp_path: Path) -> None:
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("dummy content")
    with pytest.raises(ValueError, match="not supported"):
        load_data(txt_file)


def test_load_data_completely_empty_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="is empty"):
        load_data(empty_file)


def test_load_data_header_only_file(tmp_path: Path) -> None:
    """A CSV with a header but no rows is treated as empty."""
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("PassengerId,Survived,Pclass\n")
    with pytest.raises(ValueError, match="is empty"):
        load_data(header_only)


def test_save_cleaned_data_creates_parents_and_omits_index(
    tmp_path: Path, sample_clean_df: pd.DataFrame
) -> None:
    out_file = tmp_path / "nested" / "dirs" / "cleaned.csv"
    save_cleaned_data(sample_clean_df, out_file)

    assert out_file.exists()
    reloaded = pd.read_csv(out_file)
    assert reloaded.shape == sample_clean_df.shape
    assert "Unnamed: 0" not in reloaded.columns


# 3. Validation & Imputation Tests

def test_validate_columns_valid(sample_raw_df: pd.DataFrame) -> None:
    validate_columns(sample_raw_df)


def test_validate_columns_missing_single(sample_raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns.*Age"):
        validate_columns(sample_raw_df.drop(columns=["Age"]))


def test_validate_columns_missing_multiple(sample_raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_columns(sample_raw_df.drop(columns=["Survived", "Pclass"]))
    assert "Survived" in str(exc_info.value)
    assert "Pclass" in str(exc_info.value)


def test_impute_embarked_fills_with_mode(sample_raw_df: pd.DataFrame) -> None:
    res_df = impute_embarked(sample_raw_df)
    assert res_df["Embarked"].isnull().sum() == 0
    assert res_df.loc[4, "Embarked"] == "S"   # 'S' is the mode
    assert res_df.loc[1, "Embarked"] == "C"   # existing values untouched


def test_impute_embarked_does_not_mutate_input(sample_raw_df: pd.DataFrame) -> None:
    original = sample_raw_df.copy()
    impute_embarked(sample_raw_df)
    pd.testing.assert_frame_equal(sample_raw_df, original)


def test_impute_embarked_all_missing_stays_missing() -> None:
    """With no known values there is no mode, so the column is left as is."""
    df = pd.DataFrame({"Embarked": [np.nan, np.nan]})
    res_df = impute_embarked(df)
    assert res_df["Embarked"].isnull().all()


def test_impute_age_uses_pclass_sex_group_median() -> None:
    df = pd.DataFrame({
        "Pclass": [1, 1, 1, 2, 3],
        "Sex": ["male", "male", "male", "female", "female"],
        "Age": [20.0, 30.0, np.nan, 40.0, np.nan],
    })
    res_df = impute_age(df)

    # Group (1, male) has known ages 20 and 30, so the median is 25.
    assert res_df.loc[2, "Age"] == 25.0
    # Known ages are never changed.
    assert res_df.loc[0, "Age"] == 20.0
    assert res_df.loc[1, "Age"] == 30.0
    assert res_df.loc[3, "Age"] == 40.0


def test_impute_age_fallback_uses_overall_median_of_known_ages() -> None:
    """Group (3, female) has no known ages, so the overall median is used.

    Known ages are [20, 30, 40], whose median is 30.

    This also guards against a regression: if the group-imputed value 25
    (for group (1, male)) leaked into the fallback calculation, the median
    of [20, 25, 30, 40] would be 27.5 instead of 30.
    """
    df = pd.DataFrame({
        "Pclass": [1, 1, 1, 2, 3],
        "Sex": ["male", "male", "male", "female", "female"],
        "Age": [20.0, 30.0, np.nan, 40.0, np.nan],
    })
    res_df = impute_age(df)
    assert res_df.loc[4, "Age"] == 30.0


def test_impute_age_all_missing_stays_missing() -> None:
    df = pd.DataFrame({
        "Pclass": [1, 2],
        "Sex": ["male", "female"],
        "Age": [np.nan, np.nan],
    })
    res_df = impute_age(df)
    assert res_df["Age"].isnull().all()


def test_impute_age_does_not_mutate_input(sample_raw_df: pd.DataFrame) -> None:
    original = sample_raw_df.copy()
    impute_age(sample_raw_df)
    pd.testing.assert_frame_equal(sample_raw_df, original)


def test_engineer_cabin_features(sample_raw_df: pd.DataFrame) -> None:
    res_df = engineer_cabin_features(sample_raw_df)
    assert "Cabin" not in res_df.columns
    assert res_df.loc[0, "Has_Cabin"] == 0
    assert res_df.loc[1, "Has_Cabin"] == 1
    assert res_df.loc[0, "Deck"] == "Unknown"
    assert res_df.loc[1, "Deck"] == "C"
    assert res_df.loc[5, "Deck"] == "E"


def test_engineer_cabin_features_when_all_cabins_missing() -> None:
    """A fully empty Cabin column must not crash the deck extraction."""
    df = pd.DataFrame({"Cabin": [np.nan, np.nan]})
    res_df = engineer_cabin_features(df)
    assert "Cabin" not in res_df.columns
    assert res_df["Has_Cabin"].tolist() == [0, 0]
    assert res_df["Deck"].tolist() == ["Unknown", "Unknown"]


@pytest.mark.parametrize(
    "age, expected_group",
    [
        (0, "Child (0-12)"),
        (0.5, "Child (0-12)"),       # infants must not be dropped
        (12, "Child (0-12)"),
        (13, "Teen (13-18)"),
        (18, "Teen (13-18)"),
        (19, "Young Adult (19-35)"),
        (60, "Adult (36-60)"),
        (61, "Elderly (60+)"),
    ],
)
def test_assign_age_group(age: float, expected_group: str) -> None:
    result = assign_age_group(pd.Series([age]))
    assert str(result.iloc[0]) == expected_group


def test_assign_fare_quartile_has_no_missing_values() -> None:
    fares = pd.Series([7.25, 71.28, 7.92, 53.10, 8.05, 10.0, 10.0, 8.05, 8.05, 26.0])
    result = assign_fare_quartile(fares)
    assert result.isnull().sum() == 0
    assert 1 <= len(result.cat.categories) <= 4


def test_cleaner_pipeline(sample_raw_df: pd.DataFrame) -> None:
    cleaned = cleaner(sample_raw_df)

    assert cleaned.shape[0] == 6                    # one duplicate removed
    assert cleaned.shape[1] == 13                   # 12 - Cabin + Has_Cabin + Deck
    assert cleaned.isnull().sum().sum() == 0
    assert "Has_Cabin" in cleaned.columns
    assert "Deck" in cleaned.columns
    assert "Cabin" not in cleaned.columns


def test_cleaner_does_not_mutate_input(sample_raw_df: pd.DataFrame) -> None:
    original = sample_raw_df.copy()
    cleaner(sample_raw_df)
    pd.testing.assert_frame_equal(sample_raw_df, original)


def test_cleaner_rejects_invalid_schema(sample_raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        cleaner(sample_raw_df.drop(columns=["Embarked"]))


# 4. Visualization Tests (headless)

@pytest.mark.parametrize("plot_func", PLOT_FUNCTIONS, ids=lambda f: f.__name__)
def test_plot_function_runs_and_shows_once(
    plot_func, sample_clean_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    shows = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shows.append(True))

    plot_func(sample_clean_df)

    assert len(shows) == 1
    assert plt.get_fignums() == []  # figure was closed by the plot function


# 5. Interactive CLI Menu Tests

@pytest.mark.parametrize(
    "expected_line",
    [
        "Titanic EDA Visualizer",
        "1) Overall Survival",
        "2) Gender Survival",
        "3) Class Survival",
        "4) Age Distribution",
        "5) Fare Distribution",
        "6) Multivariate (Class & Sex)",
        "7) Correlation Heatmap",
        "0) Exit",
    ],
)
def test_print_menu_lists_all_options(capsys: pytest.CaptureFixture[str], expected_line: str) -> None:
    print_menu()
    assert expected_line in capsys.readouterr().out


@pytest.mark.parametrize("choice, func_name", list(MENU_DISPATCH.items()))
def test_menu_dispatches_each_choice(
    choice: str,
    func_name: str,
    monkeypatch: pytest.MonkeyPatch,
    sample_clean_df: pd.DataFrame,
) -> None:
    calls = []
    monkeypatch.setattr(clean_data, func_name, lambda df: calls.append(df))

    inputs = iter([choice, "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    visualization_menu(sample_clean_df)

    assert len(calls) == 1
    assert calls[0] is sample_clean_df


def test_menu_strips_whitespace_around_choice(
    monkeypatch: pytest.MonkeyPatch, sample_clean_df: pd.DataFrame
) -> None:
    calls = []
    monkeypatch.setattr(clean_data, "plot_overall_survival", lambda df: calls.append(df))

    inputs = iter([" 1 ", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    visualization_menu(sample_clean_df)
    assert len(calls) == 1


def test_menu_exit_immediately(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_clean_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "0")

    visualization_menu(sample_clean_df)
    assert "Exiting visualizer" in capsys.readouterr().out


def test_menu_invalid_inputs_warn_and_continue(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sample_clean_df: pd.DataFrame,
) -> None:
    inputs = iter(["99", "abc", "", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    visualization_menu(sample_clean_df)
    captured = capsys.readouterr().out
    assert captured.count("[!] Invalid input") == 3
    assert "Exiting visualizer" in captured


# 6. End-to-End Test

def test_main_end_to_end(
    tmp_path: Path,
    sample_raw_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_raw_df.to_csv(data_dir / "train.csv", index=False)

    monkeypatch.setattr(clean_data, "BASE_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["clean_data.py"])
    monkeypatch.setattr("builtins.input", lambda _: "0")
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    clean_data.main()

    output_file = data_dir / "cleaned_train.csv"
    assert output_file.exists()

    cleaned = pd.read_csv(output_file)
    assert cleaned.shape == (6, 13)
    assert cleaned.isnull().sum().sum() == 0