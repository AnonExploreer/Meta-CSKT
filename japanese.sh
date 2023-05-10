# python train_ours.py --cfg ./experiment/japanese.yaml --update-steps 600 --name animal_japanesemacaque_l030f010 \
#      --l-threshold 0.030 --f-threshold 0.1 --human_shift japanese_human.csv --flip_shift japanese_flip.csv --total-steps 10000  \
#      --save-path ./checkpoint --seed 5   --dataset animalweb   --lambda-u 8  --student-wait-steps 0  --resume ./pretrained_model/japanese.pth
# python train_ours.py --cfg ./experiment/japanese.yaml --update-steps 600 --name animal_japanesemacaque_l025f010 --l-threshold 0.025 --f-threshold 0.1 --human_shift japanese_human.csv --flip_shift japanese_flip.csv --total-steps 5000  --save-path ./checkpoint --seed 5   --dataset animalweb   --lambda-u 8  --student-wait-steps 0  --resume ./pretrained_model/japanese.pth
# python train_ours.py --cfg ./experiment/japanese.yaml --update-steps 600 --name animal_japanesemacaque_l020f010 --l-threshold 0.020 --f-threshold 0.1 --human_shift japanese_human.csv --flip_shift japanese_flip.csv --total-steps 5000  --save-path ./checkpoint --seed 5   --dataset animalweb   --lambda-u 8  --student-wait-steps 0  --resume ./pretrained_model/japanese.pth
# python train_ours.py --cfg ./experiment/japanese.yaml --update-steps 600 --name animal_japanesemacaque_l015f010 --l-threshold 0.015 --f-threshold 0.1 --human_shift japanese_human.csv --flip_shift japanese_flip.csv --total-steps 7000  --save-path ./checkpoint --seed 5   --dataset animalweb   --lambda-u 8  --student-wait-steps 0  --resume ./pretrained_model/japanese.pth

# python train_ours.py --cfg ./experiment/japanese.yaml --update-steps 600 --name animal_japanesemacaque_lnaf010 --l-threshold 0.0 --f-threshold 0.1 --human_shift japanese_human.csv --flip_shift japanese_flip.csv --total-steps 7000  --save-path ./checkpoint --seed 5   --dataset animalweb   --lambda-u 8  --student-wait-steps 0  --resume ./pretrained_model/japanese.pth

# CUDA_VISIBLE_DEVICES=1 nohup python train_ours.py --cfg ./experiment/cercopithecidae.yaml --update-steps 600 --name animal_cercopithecidae_l050f020 --l-threshold 0.05 --f-threshold 0.2 --human_shift japanese_human.csv --flip_shift japanese_flip.csv --total-steps 7000  --save-path ./checkpoint --seed 5   --dataset animalweb   --lambda-u 8  --student-wait-steps 0  --resume ./pretrained_model/japanese.pth > cercopithecidae.out &


