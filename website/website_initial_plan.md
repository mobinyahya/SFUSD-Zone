I would like to make a website that lets community stakeholders interact with the agent in LLM/exploration/. It should have a chat bar, where users can interact with the zoning agent. However, importantly they should also be able to interact with zoning produced by the agent. So, when they take a look at the current zoning, there should be a map that overlays color coded and labelled zones. Zones are composed of block groups, and each one should be hoverable and clickable. And have a popup that has its block group id, the demographics of students in that zone, and indicate which zone it is assigned to. You can take a look in Graphic_Visualization/zone_viz.py, to see where the geo data file paths are. This is more of a data dashboard, so their will only be one page, but it is important that it allows them to effectively interact with the zoning agent and see the current solution that they are talking about with the zoning agent.

The map should be on the left, with the chat bar on the right. At the bottom, there should be graphs that show information about the current zoning, such as a plot of the demographics of each zone, or the total number of students in each zone. 

The website should be clear and load fast.

For now, I am creating a research demo that I will iterate upon later. So use the data from /home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/llm_bg_runs/recursive_metrics_flattened.csv

Notably, each solution will contain a column called path, which leads to a folder with information about that zoning. To find the data related to the final zoning for this solution, use data related to BlockGroup_0. You can find the “zone_dict” (a mapping of blocks to zones in zone_dict_BlockGroup_0.json of this folder from path). The keys of a zone_dict are ids, which can be mapped to entries in a networkx graph object that can loaded from graph_folder = f'{get_dropbox_path(config["is_local"])}/Optimization/Zones/Graphs' graph_filename = os.path.join(graph_folder, f"BlockGroup_0.pickle")
 with open(graph_filename, "rb") as f:
	base_G = pickle.load(f)

Look at Zone_Generation/Optimization/create_larger_areas.py for the structure of these graphs. I am running this on a linux research server, and for now everything can be in local host. Write your code into the website folder.

For now we are going to work on this frontend display of the data only, no connection to infra running the agent, but you should create a brief overview of how this connection would work in the future. 

Create a plan for this project and ask me for approval of it.Focus on writing lower verbosity and clear code.