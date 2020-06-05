import sys
import json

import popart

model_prefix = sys.argv[1]
with open(model_prefix + '.json', 'r') as fp:
    config = json.load(fp)

model_suffices = ['_head.onnx', '_tail.onnx']
for scale in config.get('scales'):
    model_suffices.append(f'_upsample_x{scale}.onnx')

model_paths = [model_prefix + model_suffix for model_suffix in model_suffices]
for model_path in model_paths:
    graph_transformer = popart.GraphTransformer(model_path)
    graph_transformer.convertFloatsToHalfs()
    with open(model_path, 'wb') as fp:
        fp.write(graph_transformer.getModelProto())
