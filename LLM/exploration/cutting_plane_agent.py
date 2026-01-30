import os
import pprint
from typing import Optional, Dict

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CSV_PATH = "/home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/llm_bg_runs/recursive_metrics_flattened.csv"


class MetricBounds(BaseModel):
    frl_dev_from_average: Optional[float] = None
    black_dev_from_average: Optional[float] = None
    hispanic_dev_from_average: Optional[float] = None
    white_dev_from_average: Optional[float] = None
    asian_dev_from_average: Optional[float] = None
    seat_disparity_from_average: Optional[float] = None
    avg_distance: Optional[float] = None
    boundary_cost: Optional[float] = None
    message_to_user: Optional[str] = None


def get_pareto_frontier(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Calculates the Pareto frontier for the dataframe based on metrics."""
    costs = df[metrics].values

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        # Check if any point in the dataset dominates c
        # j dominates c if all(j <= c) and any(j < c)
        dominates_c = np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
        if np.any(dominates_c):
            is_efficient[i] = False

    return df[is_efficient].copy()


class CuttingPlaneAgent:
    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # Mapping from Pydantic fields to CSV columns
        col_map = {
            'frl_dev_from_average': 'FRL',
            'black_dev_from_average': 'Ethnicity_Black_or_African_American',
            'hispanic_dev_from_average': 'Ethnicity_Hispanic/Latinx',
            'white_dev_from_average': 'Ethnicity_White',
            'asian_dev_from_average': 'Ethnicity_Asian',
            'seat_disparity_from_average': 'seat_disparity',
            'avg_distance': 'closest_school_distances',
            'boundary_cost': 'boundary_cost'
        }
        self.metrics = list(col_map.keys())

        # Rename columns to match Pydantic fields
        inv_map = {v: k for k, v in col_map.items()}
        self.df = self.df.rename(columns=inv_map)

        # Preprocess: Filter for Pareto optimal solutions across all metrics
        print(f"Initial solution space: {len(self.df)} solutions")
        self.df = get_pareto_frontier(self.df, self.metrics)
        print(f"Pareto frontier size: {len(self.df)} solutions")

        self.client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',
        )

        self.current_bounds = {}
        self.chat_history = [None]
        self.reset_filters()

        self.feasible_df = self.get_feasible_solutions()
        self.cur_representative_sol = self._get_representative_solution()

    def reset_filters(self):
        """Initialize internal state to loosest filters (max values in validation set)"""
        self.current_bounds = {}
        self.chat_history = [None]
        for metric in self.metrics:
            if metric in self.df.columns:
                self.current_bounds[metric] = float(self.df[metric].max())
            else:
                print(f"Warning: Metric column {metric} not found in CSV")
                self.current_bounds[metric] = float('inf')

    def _get_current_stats(self):
        stats = {}
        for metric in self.metrics:
            if metric in self.df.columns:
                series = self.feasible_df[metric]
                global_series = self.df[metric]
                if not series.empty:
                    stats[metric] = {
                        'global_min': float(global_series.min()),
                        'global_max': float(global_series.max()),
                        'feasible_min': float(series.min()),
                        'feasible_mean': float(series.mean()),
                        'feasible_max': float(series.max()),
                        'current_bound': self.current_bounds[metric]
                    }
        return stats

    def update_filters(self, user_input: str):
        if self.feasible_df.empty:
            return "No feasible solutions currently. Please reset filters or loosen criteria."

        if len(self.feasible_df) == 1:
            rep_sol = self.feasible_df.iloc[0]
            metrics = rep_sol[self.metrics].to_dict()
            return f"Already isolated a single solution (Index: {rep_sol.name}).\nMetrics: {metrics}"

        llm_response = self._call_llm(user_input)

        return self._process_llm_response(llm_response)

    def _call_llm(self, user_input: str) -> MetricBounds:
        self.chat_history[0] = {"role": "system", "content": self._prepare_refinement_prompt()}

        self.chat_history.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.parse(
                model='gemma3:27b',
                messages=self.chat_history,  # type: ignore
                response_format=MetricBounds
            )
            new_bounds = response.choices[0].message.parsed

            # Update history
            self.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})

            return new_bounds
        except Exception as e:
            print(f"LLM Error: {e}")
            raise

    def _prepare_refinement_prompt(self) -> str:
        stats = self._get_current_stats()
        rep_metrics = self.cur_representative_sol[
            self.metrics].to_dict() if self.cur_representative_sol is not None else {}

        print(stats)
        print(rep_metrics)

        return f"""
                    You are an expert school zoning consultant.
                    The user is responding to a zoning with the following metrics:\\
                    {rep_metrics}\\

                    Market context (feasible range for each metric):\\
                    {stats}\\
                    
                    Note that the boundary_cost is associated with compactness, with a lower value implying higher compactness.\\
                    Additionally note that lower deviations from the average for each race implies higher diversity.
        
                    Tighten the 'MetricBounds' based on their response to move closer to their ideal solution.
                    Always provide a helpful message in 'message_to_user'.
                """

    def _process_llm_response(self, new_bounds: MetricBounds) -> str:
        # 1. Update bounds from direct fields
        for metric in self.metrics:
            val = getattr(new_bounds, metric)
            if val is not None:
                self.current_bounds[metric] = val

        # 2. Always show latest representative solution after updates
        self.feasible_df = self.get_feasible_solutions()
        print(len(self.feasible_df))
        if not self.feasible_df.empty:
            self.cur_representative_sol = self._get_representative_solution()
            metrics_str = self.get_current_representative_solution_metrics()
            return f"{new_bounds.message_to_user}\n\nRepresentative Solution Metrics:\n{metrics_str}"

        return new_bounds.message_to_user

    def _get_representative_solution(self) -> Optional[pd.Series]:
        if self.feasible_df.empty: return None
        cols = self.metrics
        data = self.feasible_df[cols]

        # Simple Euclidean distance to the mean, normalized by std
        mean = data.mean()
        std = data.std()
        std[std == 0] = 1.0

        normalized_dist = ((data - mean) / std) ** 2
        idx = normalized_dist.sum(axis=1).idxmin()
        return self.feasible_df.loc[idx]

    def get_current_representative_solution_metrics(self):
        metrics_str = self.cur_representative_sol[self.metrics].to_dict()
        # format with pprint
        return pprint.pformat(metrics_str)

    def get_feasible_solutions(self) -> pd.DataFrame:
        mask = pd.Series(True, index=self.df.index)
        for metric in self.metrics:
            if metric in self.df.columns:
                mask &= (self.df[metric] <= self.current_bounds[metric])
        return self.df[mask].copy()


if __name__ == "__main__":
    agent = CuttingPlaneAgent()

    print(agent.get_current_representative_solution_metrics())
    while True:
        # first print the current represenative solutions
        user_text = input('User Input (or \'exit\'): ')
        if user_text.lower() == 'exit':
            break
        print(agent.update_filters(user_text))
