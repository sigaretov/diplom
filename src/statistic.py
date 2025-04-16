import json
import os
import copy
from ABC import *
from embedding import *

def calc_avr(stats):
        avr = {
            "avr_changed":          0, "max_changed":          0, "min_changed": 1e9,
            "avr_pnsr":             0, "max_pnsr":             0, "min_pnsr": 1e9,
            "avr_error_function":   0, "max_error_function":   0, "min_error_function": 1e9,
            "avr_side_info_size":   0, "max_side_info_size":   0, "min_side_info_size": 1e9,
            "avr_bits_overcap":     0, "max_bits_overcap":     0, "min_bits_overcap": 1e9,
            "avr_restored_percent": 0, "max_restored_percent": 0, "min_restored_percent": 1e9,
        }
        for stat in stats:
            avr["avr_changed"] += stat["changed"]
            avr["max_changed"] = max(avr["max_changed"], stat["changed"])
            avr["min_changed"] = min(avr["min_changed"], stat["changed"])
            avr["avr_pnsr"] += stat["pnsr"]
            avr["max_pnsr"] = max(avr["max_pnsr"], stat["pnsr"])
            avr["min_pnsr"] = min(avr["min_pnsr"], stat["pnsr"])
            avr["avr_error_function"] += stat["error_function"]
            avr["max_error_function"] = max(avr["max_error_function"], stat["error_function"])
            avr["min_error_function"] = min(avr["min_error_function"], stat["error_function"])
            avr["avr_side_info_size"] += stat["side_info_size"]
            avr["max_side_info_size"] = max(avr["max_side_info_size"], stat["side_info_size"])
            avr["min_side_info_size"] = min(avr["min_side_info_size"], stat["side_info_size"])
            avr["avr_bits_overcap"] += stat["bits_overcap"]
            avr["max_bits_overcap"] = max(avr["max_bits_overcap"], stat["bits_overcap"])
            avr["min_bits_overcap"] = min(avr["min_bits_overcap"], stat["bits_overcap"])
            avr["avr_restored_percent"] += stat["restored_percent"]
            avr["max_restored_percent"] = max(avr["max_restored_percent"], stat["restored_percent"])
            avr["min_restored_percent"] = min(avr["min_restored_percent"], stat["restored_percent"])
        
        avr["avr_changed"] /= len(stats)
        avr["avr_pnsr"] /= len(stats)
        avr["avr_error_function"] /= len(stats)
        avr["avr_side_info_size"] /= len(stats)
        avr["avr_bits_overcap"] /= len(stats)
        avr["avr_restored_percent"] /= len(stats)
        return avr

def aggregate_stats(stats):
    for stat in stats:
        raw = stats[stat]["raw"]
        stats[stat]["avr"] = calc_avr(raw)

def save_stats(stats, name):
    folder = './stats'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f'{folder}/{name}', 'w') as f:
        json.dump(stats, f)

def read_stats(name):
    folder = './stats'
    with open(f'{folder}/{name}', 'r') as f:
        return json.load(f)

def clear_stats():
    folder = './stats'
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))