python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 1 --width 320 --height 180 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 9 --width 240 --height 240 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 9 --width 160 --height 160 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r32c64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --batches-per-step 1 --width 320 --height 180 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r32c64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --batches-per-step 9 --width 240 --height 240 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 3 --batches-per-step 1 --width 480 --height 270 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 3 --batches-per-step 1 --width 240 --height 240 --pad-factor 2
python3 onnxport.py --model MDSR --pre_train ../../models/mdsr-x2,3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2+3 --batches-per-step 9 --width 240 --height 240 --pad-factor 1
