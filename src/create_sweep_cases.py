import json
import numpy as np

def count_decimals(number):
    number_string = str(number)
    if '.' not in number_string:
        return 0
    decimal_part = number_string.split('.')[1]
    return len(decimal_part)

def getRate(step_size):
    rate = []
    for i in np.arange(0, 1, step_size): 
        rate.append(round(i, count_decimals(step_size)))
    
    return rate

models = {'roberta': 12, 'gptj': 28, 'llama2': 32}
def getLayer(model): 
    layers = []
    assert(model in models)

    for i in range(models[model] + 1): 
        layers.append(i)
    
    return layers

lnames = ['fc_in', 'fc_out']

try:
    # Open and read the JSON file
    with open('sweep.json', 'r') as file:
        data = json.load(file)
        data["lname"] = lnames
        data["rate"] = getRate(0.01)
        data["lnum"] = getLayer('gptj')
        print(json.dumps(data, indent=4))  # Beautify the printed JSON

except FileNotFoundError:
    print("The file was not found.")

except json.JSONDecodeError:
    print("The file contains invalid JSON.")