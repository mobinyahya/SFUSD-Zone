import gc

from Zone_Generation.Optimization_CP import model

import subprocess

def run_job():
    time = 60 * 60 * 5
    zones = '9-zone-1'
    model.main(time, zones)
    gc.collect()
    zones = '13-zone-2'
    model.main(time, zones)
    gc.collect()
    zones = '18-zone-6'
    gc.collect()
    model.main(time, zones)


if __name__ == "__main__":
    # subprocess.run(['conda activate SFUSD-Zone]'])
    #
    # subprocess.run(["ls", "-l"])
    run_job()
