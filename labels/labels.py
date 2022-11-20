from collections import defaultdict
import json


def label_txt_to_dict(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = lines[3:]
    lines = [line.strip() for line in lines]


    result = {}

    current_class = ""
    for line in lines:
        if not line.isnumeric() and line != "":
            current_class = line.split(" ")[0]
        if line.isnumeric():
            result[int(line)] = current_class
    return result

def create_label_to_id(dictionary):
    result = defaultdict(lambda: [])
    for key in dictionary.keys():
        result[dictionary[key]].append(key)
    for key in result.keys():
        result[key].sort()

    return dict(result)

if __name__ == "__main__":
    # 762 is not present???
    for version in  ['7', '49', '92']:
        dict1 = label_txt_to_dict(f"labels/test{version}.cla")
        dict2 = label_txt_to_dict(f"labels/train{version}.cla")
        dict1.update(dict2)
        with open(f"labels/id_to_label_{version}.json", "w") as file:
            json.dump(dict1,file) 
        label_to_id = create_label_to_id(dict1)
        with open(f"labels/label_to_id_{version}.json", "w") as file:
            json.dump(label_to_id,file) 
    
