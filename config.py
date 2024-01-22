import json


path_config = {
    'geoinfo': '5_info.csv',
    'image': 'datasets/5'
}

# 保存配置为 JSON 文件
with open('path_config.json', 'w') as json_file:
    json.dump(path_config, json_file, indent=4)

print("configuration saved to 'path_config.json'")