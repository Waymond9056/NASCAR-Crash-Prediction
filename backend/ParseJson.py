import json
import numpy as np
import pandas as pd
class ParseJson:
    def parse_json(file_name):
        with open(file_name, "r") as file:
            json_dict = json.load(file)
            return json_dict
    def get_driver_info(file_name):
        json_dict = ParseJson.parse_json(file_name)
        number_laps = 326                                               # Add one to the actual number of laps for Lap 0
        lap_times = np.full((number_laps, 40),-1.0)
        headers = []
        
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
        return lap_time_data
    
    def get_lap_tensor(file_name, lap_number):
        pass

ParseJson.get_driver_info("backend/JsonData/2023_Spring.json")
