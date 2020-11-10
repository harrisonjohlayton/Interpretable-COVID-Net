
import math
import numpy as np

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

def read_results_file(results_file='output/eval_results/no_crop/results.txt', no_correct = 100, print_dict=False):
    fd = open(results_file, 'r')
    lines = fd.readlines()
    incorrect = []
    correct_covid = []
    correct_pneumonia = []
    correct_normal = []
    for line in lines:
        words = line.split()
        image_name = words[0]
        true_label = words[1]
        predicted_label = words[2]
        softmax_values = [float(words[3]), float(words[4]), float(words[5])]
        predicted_value = softmax_values[mapping[predicted_label]]

        result_map = {
            'image_name':image_name,
            'true_label':true_label,
            'predicted_label':predicted_label,
            'softmax_values':softmax_values,
            'predicted_value':predicted_value,
        }

        if (predicted_label == true_label):
            #find only the top 5 most correct of each class
            if (true_label == 'normal'):
                correct_array = correct_normal
            elif (true_label == 'pneumonia'):
                correct_array = correct_pneumonia
            else:
                correct_array = correct_covid

            #insert into correct position
            if (len(correct_array) < no_correct):
                correct_array.append(result_map)
            else:
                min_val = 2.0
                min_idx = -1
                for i in range(len(correct_array)):
                    result = correct_array[i]
                    if (result['predicted_value'] < min_val):
                        min_val = result['predicted_value']
                        min_idx = i
                if (predicted_value > min_val):
                    correct_array.pop(min_idx)
                    correct_array.append(result_map)
        else:
            incorrect.append(result_map)
    return_dict = dict()
    for i in range(len(correct_normal)):
        return_dict[f'normal_{i}'] = correct_normal[i]
    for i in range(len(correct_pneumonia)):
        return_dict[f'pneumonia_{i}'] = correct_pneumonia[i]
    for i in range(len(correct_covid)):
        return_dict[f'covid_{i}'] = correct_covid[i]
    for i in range(len(incorrect)):
        return_dict[f'incorrect_{i}'] = incorrect[i]

    if (print_dict):
        print(f'found {len(correct_normal)} most correct normal evals')
        print(f'found {len(correct_pneumonia)} most correct pneumonia evals')
        print(f'found {len(correct_covid)} most correct covid evals')
        print(f'found {len(incorrect)} incorrect evals')
        print(return_dict)
    return return_dict

def replace_activations(acts, firings):
    for x in range(len(acts)):
        for y in range(len(acts[x])):
            for channel in range(len(acts[x][y])):
                percentile = get_percentile(acts[x][y][channel], firings[channel])
                acts[x][y][channel] = percentile

def get_percentile(firing, firings):
    min_spot = 0
    max_spot = len(firings)-1
    while(max_spot - min_spot > 1):
        next_spot = min_spot + math.floor((max_spot - min_spot)/2)
        if (firing > firings[next_spot]):
            min_spot = next_spot
        else:
            max_spot = next_spot
    percentage = (firing - firings[min_spot]) / (firings[max_spot] - firings[min_spot])
    return 100-(min_spot + percentage)


def get_activation_mask(img, acts, channel):
    new_img = np.array(img, copy=True)
    pixels_per_neuron = len(img)/len(acts)
    for i in range(len(img)):
        for j in range(len(img)):
            x = math.floor(i/pixels_per_neuron)
            y = math.floor(j/pixels_per_neuron)
            act = (acts[x][y][channel]/100)**4

            for q in range(3):
                new_img[i][j][q] = img[i][j][q]*act

    return new_img
