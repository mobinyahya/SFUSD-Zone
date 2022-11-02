import pandas as pd
from pathlib import Path
import numpy as np


def load_student_data():
    df = pd.DataFrame()
    for year in [14, 15, 16, 16, 18, 21]:
        y = pd.read_csv(Path(f"~/SFUSD/Data/Cleaned/enrolled_{year}{year + 1}.csv").expanduser())
        df = df.append(y)
    return df


def create_programs_col(df):
    df['applied_programs'] = np.nan
    for rnd in range(1, 6):
        if f"r{rnd}_programs" in df.columns:
            df["applied_programs"] = df["applied_programs"].fillna(df[f"r{rnd}_programs"])
    df.applied_programs = df.applied_programs.apply(lambda x: eval(x) if not pd.isna(x) else [])


def create_count_column(df, program_code):
    df[f"enrolled_and_{program_code.lower()}_applied"] = df.apply(
        lambda x: sum([i == program_code for i in x["applied_programs"]]) / len(x["applied_programs"])
        if not pd.isna(x["enrolled_idschool"]) and len(x["applied_programs"]) > 0
        else 0,
        axis=1
    )


def format_table(df, program_code_list):
    cols = [f"enrolled_and_{x.lower()}_applied" for x in program_code_list]
    aa = df.groupby('idschoolattendance').sum()[cols]
    bg = df.groupby('census_blockgroup').sum()[cols]
    return aa, bg


def create_table(program_code_list, save_location):
    df = load_student_data()
    create_programs_col(df)
    for program_code in program_code_list:
        create_count_column(df, program_code)
    aa, bg = format_table(df, program_code_list)
    aa.to_csv(save_location / "language_zone_data_aa.csv")
    bg.to_csv(save_location / "language_zone_data_bg.csv")


if __name__ == "__main__":
    program_codes = ["SE", "SN", "CE", "CN", "CT", "SB", "CB"]
    save_path = Path("~/SFUSD/Data/Cleaned").expanduser()
    create_table(program_codes, save_path)
