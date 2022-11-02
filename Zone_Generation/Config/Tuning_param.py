class Tuning_param():
    def __init__(self):
        # changing levers
        self.zone_count = 6
        self.centroids_type = '6-zone-4'
        self.frl_dev = 0.2
        self.racial_dev = 0.22
        self.include_k8 = False

        # fixed levers
        self.boundary_threshold = 0.2
        self.shortage = 0.24 # this is in percentage
        self.overage = 0.28 # this is in percentage
        self.all_cap_shortage = 0.21 # this is in percentage
        self.population_type = "All"
        self.max_distance = 8
        self.path = "/Users/mobin/SFUSD/Visualization_Tool_Data/"
        # self.path = "/Users/mobin/Mobin-Trial\Dropbox/Mobin\YahyazadehJeloudar/SFUSD/Optimization/Zones/Visualization_Tool/"
        self.move_SpEd = False

        # unused metrics
        self.balance = 500
        self.aalpi_dev = 0.15
        self.include_sch_qlty_balance = False
        self.lbscqlty = 0.75
        self.HOCidx1_dev = 0.2




def compute_name(param):
    name = "_Zoning"
    # check frl deviation
    if param.frl_dev == 0.15:
        name += "_1"
    else:
        name += "_2"
    # check racial deviation
    if param.racial_dev == 0.15:
        name += "_1"
    else:
        name += "_2"
    # check number of zones
    if param.zone_count == 6:
        name += "_1"
    elif param.zone_count == 7:
        name += "_2"
    else:
        name += "_3"
    # check if we are enforcing school quality balance
    if param.include_sch_qlty_balance == True:
        name += "_1"
    else:
        name += "_2"
    # check if K-8 schools are included
    if param.include_k8 == True:
        name += "_1"
    else:
        name += "_2"
    # add which centroid group was selected
    name += "_centroids " + str(param.centroids_type)
    return name

