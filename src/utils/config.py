import yaml

class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def set(self, key, value):
        self._config[key] = value

    def save(self, config_file):
        with open(config_file, "w") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)
