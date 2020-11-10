from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts

class CovidNetB(Model):
    model_path = 'models/B/lucid/covid_net.pb'
    #none of these
    # labels_path = None
    # synsets_path = None
    # dataset = None
    image_shape = [480,480,3]
    image_value_range = [0, 1]
    input_name = "input_1"

CovidNetB.layers = _layers_from_list_of_dicts(CovidNetB(), [
    {'tags': ['conv'], 'name': 'conv5_block3_out/add', 'depth':2048},
    {'tags': ['conv'], 'name': 'conv5_block2_out/add', 'depth':2048},
    {'tags': ['conv'], 'name': 'conv5_block1_out/add', 'depth':2048},
    {'tags': ['conv'], 'name': 'conv4_block6_out/add', 'depth':824},
    {'tags': ['conv'], 'name': 'conv4_block5_out/add', 'depth':824},
    {'tags': ['conv'], 'name': 'conv4_block4_out/add', 'depth':824},
    {'tags': ['conv'], 'name': 'conv4_block3_out/add', 'depth':824},
    {'tags': ['conv'], 'name': 'conv4_block2_out/add', 'depth':824},
    {'tags': ['conv'], 'name': 'conv4_block1_out/add', 'depth':824},
    {'tags': ['conv'], 'name': 'conv3_block4_out/add', 'depth':416},
    {'tags': ['conv'], 'name': 'conv3_block3_out/add', 'depth':416},
    {'tags': ['conv'], 'name': 'conv3_block2_out/add', 'depth':416},
    {'tags': ['conv'], 'name': 'conv3_block1_out/add', 'depth':416},
    {'tags': ['conv'], 'name': 'conv2_block3_out/add', 'depth':200},
    {'tags': ['conv'], 'name': 'conv2_block2_out/add', 'depth':200},
    {'tags': ['conv'], 'name': 'conv2_block1_out/add', 'depth':200},
    {'tags': ['dense'], 'name': 'norm_dense_1/Softmax', 'depth':3},
])