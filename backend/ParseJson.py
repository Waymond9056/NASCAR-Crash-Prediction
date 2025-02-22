import json
import numpy as np
import pandas as pd
class ParseJson:
    def parse_json(file_name):
        number_laps = 326
        lap_times = np.full((number_laps, 40),-1.0)
        headers = []

        
        with open(file_name, "r") as file:
            json_dict = json.load(file)
            for i in range(len(json_dict["laps"])):
                driver_info = json_dict["laps"][i]
                driver_laps = driver_info["Laps"]
                headers.append(driver_info["Number"])
                for ii in range(0, len(driver_laps)):                   # Change to 0 when input offsets are done
                    lap_times[ii][i] = driver_laps[ii]["LapTime"]

        for i in range(40 - len(headers)):
            headers.append(-1)
            
        lap_time_data = pd.DataFrame(lap_times, columns=headers)
        lap_time_data.to_csv("output.csv", index = False)

ParseJson.parse_json("backend/JsonData/2024_Spring.json")
