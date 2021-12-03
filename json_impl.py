# Dependencies
import json
import config

# Database file name
file_name = config.DATABASE_FILE_NAME

def json_wrt_data(data):
    with open(file_name, 'w', encoding="utf-8") as f:
        json.dump(data, f)

# Get data
def json_get_data():
    data = {}

    # Getting data
    with open(file_name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data