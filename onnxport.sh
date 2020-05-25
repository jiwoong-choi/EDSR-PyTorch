# models without padding
python3 src/onnxport.py --pre_train ../models/model_16x32.pt --cpu --n_resblocks 16 --n_feats 32 --scale 2  --width 40 --height 40
python3 src/onnxport.py --pre_train ../models/model_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2  --width 40 --height 40 --conv-mem-portion 0.26
python3 src/onnxport.py --pre_train ../models/model_32x64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2  --width 40 --height 40 --conv-mem-portion 0.26

# models with padding 1
python3 src/onnxport.py --pre_train ../models/model_16x32.pt --cpu --n_resblocks 16 --n_feats 32 --scale 2  --width 40 --height 40 --pad 1
python3 src/onnxport.py --pre_train ../models/model_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2  --width 40 --height 40 --conv-mem-portion 0.26 --pad 1
python3 src/onnxport.py --pre_train ../models/model_32x64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2  --width 40 --height 40 --conv-mem-portion 0.26 --pad 1

# models with padding 2
python3 src/onnxport.py --pre_train ../models/model_16x32.pt --cpu --n_resblocks 16 --n_feats 32 --scale 2  --width 40 --height 40 --pad 2
python3 src/onnxport.py --pre_train ../models/model_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2  --width 40 --height 40 --conv-mem-portion 0.26 --pad 2
python3 src/onnxport.py --pre_train ../models/model_32x64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2  --width 40 --height 40 --conv-mem-portion 0.26 --pad 2
