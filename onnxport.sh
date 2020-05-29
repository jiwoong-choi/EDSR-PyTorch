python3 src/onnxport.py --pre_train ../models/immature/model_16x32.pt --cpu --n_resblocks 16 --n_feats 32 --scale 2 --conv-mem-portion 0.16 --batches-per-step 9 --width 120 --height 120 --pad 2
python3 src/onnxport.py --pre_train ../models/immature/model_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --conv-mem-portion 0.16 --batches-per-step 9 --width 120 --height 120 --pad 2
python3 src/onnxport.py --pre_train ../models/immature/model_32x64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --conv-mem-portion 0.16 --batches-per-step 9 --width 120 --height 120 --pad 2
python3 src/onnxport.py --pre_train ../models/baseline/model_x2_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --conv-mem-portion 0.16 --batches-per-step 9 --width 120 --height 120 --pad 2
python3 src/onnxport.py --pre_train ../models/baseline/model_x3_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 3 --conv-mem-portion 0.16 --batches-per-step 9 --width 80 --height 80 --pad 2
