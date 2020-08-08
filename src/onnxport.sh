# Paper models
python3 onnxport.py --model EDSR --pre_train download --cpu --res_scale 0.1 --n_resblocks 32 --n_feats 256 --scale 2 --batches-per-step 9 --width 80 --height 80 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train download --cpu --res_scale 0.1 --n_resblocks 32 --n_feats 256 --scale 3 --batches-per-step 9 --width 120 --height 120 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/standard-x2r32c256.pt --cpu --res_scale 0.1 --n_resblocks 32 --n_feats 256 --scale 2 --batches-per-step 9 --width 80 --height 80 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/standard-x3r32c256.pt --cpu --res_scale 0.1 --n_resblocks 32 --n_feats 256 --scale 3 --batches-per-step 9 --width 120 --height 120 --pad-factor 2

# 1080p -> 4k
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 9 --width 240 --height 240 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r32c64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --batches-per-step 9 --width 240 --height 240 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 6 --width 240 --height 360 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r32c64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --batches-per-step 6 --width 240 --height 360 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 5 --width 240 --height 432 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r32c64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --batches-per-step 5 --width 240 --height 432 --pad-factor 2

# 720p -> 4k
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 3 --batches-per-step 9 --width 240 --height 240 --pad-factor 2

# 720p -> 1440p & 1080p -> 4k
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 1 --width 320 --height 180 --pad-factor 2
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r32c64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --batches-per-step 1 --width 320 --height 180 --pad-factor 2

# 720p -> 4k & 1080p -> 3240p
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 3 --batches-per-step 1 --width 480 --height 270 --pad-factor 2

# 540p -> 4k
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x4-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 4 --batches-per-step 9 --width 240 --height 240 --pad-factor 2

# 640x480 -> 1280x960
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x2-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --batches-per-step 12 --width 80 --height 80 --pad-factor 2

# 640x480 -> 1920x1440
python3 onnxport.py --model EDSR --pre_train ../../models/edsr-x3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 3 --batches-per-step 12 --width 120 --height 120 --pad-factor 2

# 720p -> 4k & 1080p -> 4k
python3 onnxport.py --model MDSR --pre_train ../../models/mdsr-x2,3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2+3 --batches-per-step 9 --width 240 --height 240 --pad-factor 1
python3 onnxport.py --model MDSR --pre_train ../../models/mdsr-x2,3-r16c64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2+3 --batches-per-step 6 --width 240 --height 360 --pad-factor 1
