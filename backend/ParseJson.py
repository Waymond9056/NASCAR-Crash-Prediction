import json
import numpy as np
import pandas as pd
import torch
class ParseJson:
    @staticmethod
    def parse_json(file_name):
        with open(file_name, "r") as file:
            json_dict = json.load(file)
            return json_dict
    
    @staticmethod
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
    
    @staticmethod
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
                input_np[i][0] = 1.0
                input_np[i][1] = 1.0
                continue

            driver_number = driver_laps[dict_index]["Number"]            

            dict_before_index = dict_index_from_position[i - 1]
            driver_before_number = driver_laps[dict_before_index]["Number"]

            input_np[i][0] = curr_time.at[driver_number] - curr_time.at[driver_before_number]
            input_np[i][1] = lap_before_time.at[driver_number] - lap_before_time.at[driver_before_number]

            if input_np[i][0] > 5:
                input_np[i][0] = 1
                input_np[i][1] = 1

        return input_np
    
    @staticmethod
    def get_lap_history(file_name, lap_number):
        ret = []
        for i in range(1, lap_number + 1):
            ret.append(ParseJson.get_lap_info(file_name, i))
        return torch.Tensor(ret)

    @staticmethod
    def get_crash_laps(file_name):
        json_dict = ParseJson.parse_json(file_name)
        flag_info = json_dict["flags"]
        crash_laps = []
        caution_laps = []
        green_laps = []

        flag_status = 1
        for i in range(len(flag_info)):
            lap = flag_info[i]
            if lap["FlagState"] == 2:
                caution_laps.append(i)
                if flag_status == 1:
                    flag_status = 2
                    if i != 60 and i != 61 and i != 160 and i != 161:                    # Don't take into account stage breaks
                        for ii in range(i - 5, i):
                            if ii > 0 and not ii in caution_laps:
                                crash_laps.append(ii)
            if lap["FlagState"] == 1:
                flag_status = 1
        for i in range(1, len(flag_info)):
            if not i in crash_laps and not i in caution_laps:
                green_laps.append(i)
        return crash_laps, green_laps, caution_laps

# ParseJson.get_crash_laps("backend/JsonData/2024_Fall.json")

# print(ParseJson.get_crash_laps("backend/JsonData/2024_Fall.json"))
