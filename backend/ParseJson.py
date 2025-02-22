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
    
    def get_lap_info(file_name, lap_number):
        json_dict = ParseJson.parse_json(file_name)
        driver_laps = json_dict["laps"]

        # Store index within dict list based on running order
        # If Joey Logano (22) is first in the drivers order and is running fifth then
        # dict_index_from_position[4] = 0

        dict_index_from_position = [-1] * 40
        for i in range(len(driver_laps)):
            lap_info = driver_laps[i]["Laps"]
            if lap_number >= len(lap_info):         # Check this condition
                continue
            lap = lap_info[lap_number]
            running_position = lap["RunningPos"]
            dict_index_from_position[running_position - 1] = i
        print(dict_index_from_position)
        
        # Generate total time until this point

        lap_time_data = ParseJson.get_driver_info(file_name)
        curr_time = lap_time_data.loc[0:lap_number].sum()
        lap_before_time = lap_time_data.loc[0:lap_number - 1].sum()

        # Handle lap down cars

        # NEED TO HANDLE LAP DOWNS HERE

        # Generate an input tensor from this running order

        input_np = np.full((40,2),-1.0)
        input_np[0][0] = 0
        input_np[0][1] = 0
        for i in range(1, len(dict_index_from_position)):
            dict_index = dict_index_from_position[i]
            if dict_index == -1:
                input_np[i][0] = -1.0
                input_np[i][1] = -1.0
                print(input_np[i][0])
                continue

            driver_number = driver_laps[dict_index]["Number"]
            print(driver_number)
            

            dict_before_index = dict_index_from_position[i - 1]
            driver_before_number = driver_laps[dict_before_index]["Number"]

            input_np[i][0] = curr_time.at[driver_number] - curr_time.at[driver_before_number]
            input_np[i][1] = lap_before_time.at[driver_number] - lap_before_time.at[driver_before_number]
        print(input_np)


        


ParseJson.get_lap_info("backend/JsonData/2023_Spring.json", 5)
