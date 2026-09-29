from dotenv import load_dotenv
load_dotenv()

import yaml

def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data
 