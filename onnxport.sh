python3 src/onnxport.py --pre_train ../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --conv-mem-portion 0.16 --batches-per-step 9 --width 120 --height 120 --pad 2
python3 src/onnxport.py --pre_train ../models/edsr-x2-r32c64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --conv-mem-portion 0.16 --batches-per-step 9 --width 120 --height 120 --pad 2
