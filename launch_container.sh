DATADIRLOCAL="/data/xxx"
DATADIRLOCAL0="/data/xxx"
DATADIRLOCAL1="/datasets/240P"
DATADIRLOCAL2="/data/xxx"
DATADIRLOCAL3="/data/xxx"
DATADIRLOCAL4="/data/xxx"

if [ -z $CUDA_VISIBLE_DEVICES ]; then
   CUDA_VISIBLE_DEVICES='all'
fi


docker run --gpus device=$CUDA_VISIBLE_DEVICES --ipc=host --rm -it --name tevis \
   --mount src=$(pwd),dst=/scripts-translator,type=bind \
   --mount src=$DATADIRLOCAL,dst=/blob_mount,type=bind \
   --mount src=$DATADIRLOCAL0,dst=/blob_mount_model,type=bind,readonly \
   --mount src=$DATADIRLOCAL1,dst=/blob_mount_movienet,type=bind,readonly \
   --mount src=$DATADIRLOCAL2,dst=/blob_mount_data_yc,type=bind,readonly \
   --mount src=$DATADIRLOCAL3,dst=/blob_mount_new,type=bind \
   --mount src=$DATADIRLOCAL4,dst=/blob_mount_save,type=bind \
   -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
   -w /scripts-translator ycsun1972/azureml_docker:horovod_deepspeed_v2 \
   bash -c "source /scripts-translator/setup.sh && bash"