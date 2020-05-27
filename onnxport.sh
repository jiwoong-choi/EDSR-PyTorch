python3 -c "for i in range(21): print(f'{(0.1 + i * 0.01):.2f}')" | \
while read MEMORY_PORTION; \
do (python3 src/onnxport.py --pre_train ../models/immature/model_16x32.pt --cpu --n_resblocks 16 --n_feats 32 --scale 2 --conv-mem-portion ${MEMORY_PORTION} --batches-per-step $1 --width $2 --height $3 --pad $4 && \
python3 src/onnxport.py --pre_train ../models/immature/model_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --conv-mem-portion ${MEMORY_PORTION} --batches-per-step $1 --width $2 --height $3 --pad $4 && \
python3 src/onnxport.py --pre_train ../models/immature/model_32x64.pt --cpu --n_resblocks 32 --n_feats 64 --scale 2 --conv-mem-portion ${MEMORY_PORTION} --batches-per-step $1 --width $2 --height $3 --pad $4 && \
python3 src/onnxport.py --pre_train ../models/baseline/model_x2_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 2 --conv-mem-portion ${MEMORY_PORTION} --batches-per-step $1 --width $2 --height $3 --pad $4 && \
python3 src/onnxport.py --pre_train ../models/baseline/model_x3_16x64.pt --cpu --n_resblocks 16 --n_feats 64 --scale 3 --conv-mem-portion ${MEMORY_PORTION} --batches-per-step $1 --width $2 --height $3 --pad $4);
done
