python3 src/onnxport.py --pre_train ../models/model_16x32.pt --cpu --n_resblocks 16 --n_feats 32 --scale 2
python3 src/onnxport.py --pre_train ../models/model_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2
python3 src/onnxport.py --pre_train ../models/model_32x64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2
