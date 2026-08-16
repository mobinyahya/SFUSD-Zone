import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

if __package__:
    from .student_assignment.configerator import Configerator
    from .student_assignment.market_generator.school_choice_market import (
        SchoolChoiceMarket,
    )
else:
    from student_assignment.configerator import Configerator
    from student_assignment.market_generator.school_choice_market import (
        SchoolChoiceMarket,
    )


class ConvertEstimates:
    def __init__(
        self, model_path: str, features_path: str, distance_weight: float
    ) -> None:
        self._model_path = model_path
        self._features_path = features_path
        self._distance_weight = distance_weight

        self._configurator = Configerator()
        self._market = SchoolChoiceMarket(configurator=self._configurator)
        self._config = self._market.config

        self._grade = self._config["grade"]
        self._year = self._config["year"]
        (
            self._weights,
            self._features_df,
            self._features,
            self._students,
            self._programs,
        ) = self._load_data()

    def _load_weights(self, weights_df: pd.DataFrame) -> tuple[np.ndarray, list]:
        independent_features = list(weights_df.index)
        weights = weights_df["coefficient"].to_numpy(dtype=float, copy=True)

        if self._distance_weight is not None:
            distance_indices = [
                index
                for index, feature in enumerate(independent_features)
                if str(feature).strip().lower() == "distance"
            ]
            if len(distance_indices) != 1:
                raise ValueError(
                    "Cannot override the distance coefficient: weights.csv must "
                    "contain exactly one feature named 'distance'; found "
                    f"{len(distance_indices)} in {independent_features}."
                )
            weights[distance_indices[0]] = self._distance_weight

        return weights, independent_features

    def _load_data(self):
        weights_path = os.path.join(self._model_path, "weights.csv")
        weights_df = pd.read_csv(weights_path, index_col=0)
        weights, independent_features = self._load_weights(weights_df)

        features_df = pd.read_csv(self._features_path)
        print(weights_path)
        print(self._features_path)
        print(features_df.shape)
        features_df["student_number"] = features_df["studentno"].str.split("-").str[1]

        programs = self._market.programs
        students = self._market.students
        self._student_records = students.student_data.reset_index()[["studentno"]]

        features_df = features_df[
            features_df["student_number"].astype(int).isin(students.student_data.index)
        ]
        features = features_df[independent_features].fillna(0).to_numpy()

        return (
            weights,
            features_df,
            features,
            students,
            programs,
        )

    def _get_program_type_eligibility_matrix(self):
        eligibility_map = self._students.get_qualified_programs_dict()
        program_type_idxs = self._programs.program_type_to_indices
        eligible = np.ones((self._students.n, self._programs.num_programs)) * -np.inf
        for student_idx, studentno in self._students.idx2studentno.items():
            for program_type in eligibility_map[studentno]:
                if program_type in self._programs.program_df["program_type"].unique():
                    eligible[
                        student_idx,
                        [x - 1 for x in program_type_idxs[program_type]],
                    ] = 0
        return eligible

    def _build_initial_estimate(self):
        if self._distance_weight is None:
            print("Using the model coefficients")
        else:
            print(f"Using distance coefficient {self._distance_weight}")
        estimates = self._features.dot(self._weights)
        print(self._weights)
        estimates_df = pd.DataFrame(
            zip(
                self._features_df.studentno,
                self._features_df.program_id,
                estimates,
            ),
            columns=["studentno", "program_id", "utility"],
        )
        estimates_df = estimates_df.pivot_table(
            values="utility", columns="program_id", index="studentno"
        )
        new_order = [
            f"{self._year}{self._year + 1}-{self._students.idx2studentno[i]}"
            for i in sorted(self._students.idx2studentno.keys())
        ]
        estimates_df = estimates_df.reindex(new_order)
        return estimates_df

    def _reorder_columns(self, estimates_df):
        estimates_df.rename(columns=self._programs.indices, inplace=True)
        sorted_columns = sorted(estimates_df.columns, key=int)
        estimates_df = estimates_df[sorted_columns]
        print(estimates_df)
        self.prog_to_idx = dict(zip(sorted_columns, range(len(sorted_columns))))
        estimates_df.rename(columns=self._programs.codes, inplace=True)
        return estimates_df

    def build_estimates(self):
        estimates_df = self._build_initial_estimate()
        estimates_df = self._reorder_columns(estimates_df)
        print(
            estimates_df.shape,
            self._get_program_type_eligibility_matrix().shape,
        )
        estimates_df = estimates_df + self._get_program_type_eligibility_matrix()
        assert estimates_df.values.shape == (
            self._students.n,
            self._programs.num_programs,
        )
        if self._distance_weight is None:
            save_path = os.path.join(self._model_path, "estimates.npy")
        else:
            save_path = os.path.join(
                self._model_path, f"estimates_{self._distance_weight}.npy"
            )
        student_data = self._student_records
        estimates_df = estimates_df.reset_index()
        estimates_df["studentno"] = estimates_df["studentno"].apply(
            lambda x: int(x.split("-")[1])
        )
        estimates_df = student_data[["studentno"]].merge(
            estimates_df, on="studentno", how="left"
        )
        estimates_df["studentno"] = estimates_df["studentno"].apply(
            lambda x: f"{self._year}{self._year + 1}-{x}"
        )
        estimates_df = estimates_df.set_index("studentno")

        np.save(save_path, estimates_df.values)
        estimates_df.to_csv(save_path.replace(".npy", ".csv"))


@click.command()
@click.argument("model_path")
@click.argument(
    "features_path"
)  # TODO: generate features file without loading (need codes from SFUSD-Choice)
@click.option("--distance_weight", type=float, help="Optional distance weight")
def create_simulator_input(model_path, features_path, distance_weight):
    ce = ConvertEstimates(model_path, features_path, distance_weight)
    ce.build_estimates()


if __name__ == "__main__":
    create_simulator_input()
