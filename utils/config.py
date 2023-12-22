import yaml, json
import os, io

class Config(dict):
  def __init__(self, path=None, **elements):
    """Initiate a config object, where specified elements override the default config loaded"""
    super(Config, self).__init__(self._try_load_path(path))
    self.update(**elements)

  def _load_json(self, json_path):
    with io.open(json_path, "r", encoding="utf-8") as jf:
      return json.load(jf)

  def _load_yaml(self, yaml_path):
    with io.open(yaml_path, "r", encoding="utf-8") as yf:
      return yaml.safe_load(yf.read())

  def _try_load_path(self, path):
    assert isinstance(path, str), "Basic Config class can only support a single file path (str), but instead is {}({})".format(path, type(path))
    assert os.path.isfile(path), "Config file {:s} does not exist".format(path)
    extension = os.path.splitext(path)[-1]
    if(extension == ".json"):
      return self._load_json(path)
    elif(extension == ".yml" or extension == ".yaml"):
      return self._load_yaml(path)
    else:
      raise ValueError("Unrecognized extension ({:s}) from file {:s}".format(extension, path))

  @property
  def opt(self):
    """Backward compatibility to original. Remove once finished."""
    return self


