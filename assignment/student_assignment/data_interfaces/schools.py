"""Created 7/25/20.

@author Irene Lo

Class containing school data for market simulator
"""

import pandas as pd

from .programs import Programs


class Schools:
    def __init__(self, school_data_file: str | pd.DataFrame, programs: Programs):
        if isinstance(school_data_file, pd.DataFrame):
            init_df = school_data_file.copy()
        else:
            init_df = pd.read_csv(school_data_file, index_col=0)
        init_df = init_df.loc[
            :, ~init_df.columns.astype(str).str.startswith("Unnamed:")
        ]
        self.school_df = self._calc_attendance_area(init_df)

    def _calc_attendance_area(self, school_data: pd.DataFrame) -> pd.DataFrame():
        """Add a column mapping from each school to attendance area.

        Contains own id if an attendance area school; else 0 if citywide school.

        Args:
            school_data (pd.DataFrame): dataframe with school data

        Returns:
            pd.DataFrame: dataframe with school data and attendance area column
        """
        df = school_data.copy()
        if "school_id" not in df.columns:
            df.reset_index(level=0, inplace=True)
        elif df.index.name == "school_id":
            df.reset_index(drop=True, inplace=True)
        df["attendance_area"] = df.school_id.copy()
        df.loc[df.category == "Citywide", "attendance_area"] = 0
        df = df.set_index("school_id")
        return df

    @property
    def citywide_schools(self) -> list:
        """Get list of citywide schools.

        Returns:
            list: list of citywide schools
        """
        return self.school_df[self.school_df.category == "Citywide"].index.to_list()
