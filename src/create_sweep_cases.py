import json
import numpy as np

def count_decimals(number):
    number_string = str(number)
    if '.' not in number_string:
        return 0
    decimal_part = number_string.split('.')[1]
    return len(decimal_part)

def getRate(start, end, step_size):
    rate = []
    for i in np.arange(start, end, step_size): 
        rate.append(round(i, count_decimals(step_size)))
    
    return rate

# these are the number of layers for each model
models = {'roberta': 12, 'gptj': 28, 'llama2': 32}
def getLayer(model): 
    layers = []
    assert(model in models)

    for i in range(models[model] - 5, models[model] + 1): 
        layers.append(i)
    
    return layers

lnames = ['fc_in', 'fc_out']
use_quality = [True, False]

try:
    # Open and read the JSON file
    with open('sweep.json', 'w+') as file:
        data = {}
        data["lname"] = lnames
        data["rate"] = getRate(9.7, 9.9, 0.1)
        data["lnum"] = getLayer('gptj')
        data["use_quality"] = use_quality
        json_output = json.dumps(data, indent=4)
        print(json_output) 
        file.write(json_output)
    

except FileNotFoundError:
    print("The file was not found.")

except json.JSONDecodeError:
    print("The file contains invalid JSON.")