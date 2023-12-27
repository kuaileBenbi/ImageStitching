import glob, os
import pandas as pd

info_path = "datasets/info/2022-04-04/"
info_path_filenames = glob.glob(info_path + "*.csv")
info_path_filenames.sort()
# print(info_path_filenames)

new_file_name = info_path + "date0405_10h10m51s_10h13m31s.csv"

srcdata_time_ = []
srcdata_longtitude_ = []
srcdata_latitude_ = []

for name in info_path_filenames:
    csv_name = os.path.split(name)[1]
    if csv_name[7] == "5" and csv_name[9:11] == "10":
        cur_dataframe = pd.read_csv(name)
        srcdata_time_.extend(cur_dataframe["时间"])
        srcdata_longtitude_.extend(cur_dataframe["相机经度"])
        srcdata_latitude_.extend(cur_dataframe["相机纬度"])
    
new_dataframe = pd.DataFrame(columns=["time", "camera_longtitude", "camera_latitude"])

new_dataframe["time"] = srcdata_time_[::10]
new_dataframe["camera_longtitude"] = srcdata_longtitude_[::10]
new_dataframe["camera_latitude"] = srcdata_latitude_[::10]

new_dataframe.to_csv(new_file_name, index=False)

print("Convert done!")
