import json


class ReferencedDict(dict):
    def __getitem__(self, item):
        to_return = dict.__getitem__(self, item)
        if isinstance(to_return, str):
            to_return = to_return % self

        return to_return


def load_json_file_to_my_dict(conf_file_path):
    with open(conf_file_path) as json_file:
        conf_dict = json.load(json_file)
        return ReferencedDict(conf_dict)
