import copy
import pprint

import yaml
from openai import OpenAI
from pydantic import BaseModel

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import compute_zone_deviations
from Zone_Generation.Optimization.design_zones import Compute_Name
from Zone_Generation.Optimization.recursive_zoning import recursive_zone_supervised


class OptimizationConfig(BaseModel):
    frl_dev: float
    racial_dev: float


history = [
    {
        "role": "system",
        "content": "You are an expert in school zoning optimization. "
                   "Given the current demographic deviations in school zones and user feedback on desired adjustments, "
                   "your task is to suggest new target values for free/reduced lunch (FRL) deviation and racial "
                   "deviation to better align with user preferences. Be careful to not make these bounds too tight to "
                   "avoid infeasibility in zoning solutions. You may need to relax one bound, or keep it the same to make the other feasible"
    },
]


class LeverPullingAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',  # required, but unused
        )

    def query(self, history) -> dict:
        response = self.client.chat.completions.parse(
            model='gemma3:27b',
            messages=history,
            response_format=OptimizationConfig
        )
        return response.choices[0].message.parsed




if __name__ == "__main__":
    agent = LeverPullingAgent()

    with open("../Config/config.yaml", "r") as f:
        og_config = yaml.safe_load(f)

    name = Compute_Name(og_config)
    print("name: ", name)
    cur_racial_dev, cur_frl_dev = og_config["racial_dev"], og_config["frl_dev"]
    while True:
        cur_config = copy.deepcopy(og_config)
        cur_config["racial_dev"] = cur_racial_dev
        cur_config["frl_dev"] = cur_frl_dev
        print(cur_config)
        output, total_time = recursive_zone_supervised(cur_config)

        demographics = output.get_zone_demographics()

        print("*" * 20)
        # print demographics with pprint
        formatted_demos = pprint.pformat(compute_zone_deviations(demographics))
        print(formatted_demos)
        print("*" * 20)

        zv = ZoneVisualizer(output.dz.level, og_config['is_local'])
        zv.zones_from_dict(output.zone_dict, label=False)

        user_input = input("How would you like to have the zones adjusted?")

        history.append({
            "role": "user",
            "content": f"The demographic information for the current zoning is: {formatted_demos}\n:"
                       f"Which uses the following values for deviations - racial_dev: {cur_racial_dev}, frl_dev: {cur_frl_dev}.\n"
                       f"The user would like to adjust the zones as follows: {user_input}. "
                       f"Based on this, provide new values for racial_dev and frl_dev to align with the user's request. "

        })

        response = agent.query(history)
        print('*' * 20)
        print(response)
        print('*' * 20)
        history.append({
            "role": "assistant",
            "content": f"Based on the user's feedback, the new suggested values are - "
                       f"racial_dev: {response.racial_dev}, frl_dev: {response.frl_dev}."
        })
        cur_racial_dev, cur_frl_dev = response.racial_dev, response.frl_dev
