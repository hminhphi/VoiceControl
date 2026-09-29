from dotenv import load_dotenv
load_dotenv()
 
import yaml
from pathlib import Path
 
def read_config(path: str = 'config/config.yaml') -> dict:
    try:
        # Resolve path relative to the current file's directory if the path isn't absolute
        base_dir = Path(__file__).resolve().parent
        config_file = base_dir / path
       
        with open(config_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data or {}
    except Exception as e:
        print(f"Error reading config from {path}: {e}")
        return {}