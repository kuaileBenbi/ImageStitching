import json

# 定义相机和平台的姿态参数
camera_config = {
    "camera_position": {
        "latitude": 19.19247651,
        "longitude": 110.49602631,
        "altitude": 19038
    },
    "aircraft_angles": {
        "roll": 3.93,
        "pitch": -2.69,
        "yaw": 15.64
    },
    "pod_angles": {
        "azimuth": 0,
        "elevation": -90
    },
    "camera_specs": {
        "focal_length_mm": 40,
        "sensor_size_mm": [16, 12.8]
    }
}

# 保存配置为 JSON 文件
with open('camera_config.json', 'w') as json_file:
    json.dump(camera_config, json_file, indent=4)

print("Camera configuration saved to 'camera_config.json'")
