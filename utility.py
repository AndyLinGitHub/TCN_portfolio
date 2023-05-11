import os
import pickle
import datetime

def add_arguments(configs, parser, parent=""):
    for key, value in configs.items():
        if isinstance(value, dict):
            add_arguments(value, parser, parent + key + ".")
        else:
            parser.add_argument('--' + parent + key, help=key, default=None, type=type(value))

def nested_dict_update(value, nested_dict, keys, indent=0):
    temp = nested_dict[keys[indent]]
    if isinstance(temp, dict):
        nested_dict_update(value, temp, keys, indent+1)
    else:
        nested_dict[keys[indent]] = value

def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + str(key) + ":")
            print_dict(value, indent + 4)
        else:
            print(" " * indent + str(key) + ": " + str(value))

def dump_result(configs, result):
    output_dict = {}
    output_dict["configs"] = configs
    output_dict["result"] = result

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = configs["portfolio_config"]["model"] + f"_{timestamp}.pickle"

    if not os.path.exists(configs["setting"]["save_dir"]):
        os.mkdir(configs["setting"]["save_dir"])

    save_path = os.path.join(configs["setting"]["save_dir"], filename)
    with open(save_path, "wb") as f:
        print(save_path)
        pickle.dump(output_dict, f)

def load_result(path):
    with open(path, 'rb') as f:
        ouput_dict = pickle.load(f)

    return ouput_dict