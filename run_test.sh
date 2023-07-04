CUDA_VISIBLE_DEVICES=7 python src/run_train.py --distributed --blob_mount_dir /blob_mount --cfg src/configs/final_test.yaml
cd src/tasks
python run_task_ordering.py /blob_mount_save/tevis_test/res/00001.txt