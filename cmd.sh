# Train 80 img model
# python train_ours.py --cfg ./experiment/80.yaml --update-steps 600 --name animalweb_80img --l-threshold 0.050 --f-threshold 0.2 --human_shift human_80.csv --flip_shift flip_80.csv --total-steps 10000  --save-path ./checkpoint --seed 5   --dataset animalweb   --lambda-u 8  --student-wait-steps 0  --resume ./pretrained_model/80img_pretrain.pth

# Test 80 img for known and unknown setting
# python test.py --cfg ./experiment/test_80_known.yaml --seed 5 --name test_80_known --dataset animalweb --evaluate --resume checkpoint/animalweb_80img_best.pth.tar
# python test.py --cfg ./experiment/test_80_unknown.yaml --seed 5 --name test_80_unknown --dataset animalweb --evaluate --resume checkpoint/animalweb_80img_best.pth.tar

# Test 40 img for known and unknown setting
python test.py --cfg ./experiment/test_40_known.yaml --seed 5 --name test_40_known --dataset animalweb --evaluate --resume checkpoint/animalweb_40img_best.tar
python test.py --cfg ./experiment/test_40_unknown.yaml --seed 5 --name test_40_unknown --dataset animalweb --evaluate --resume checkpoint/animalweb_40img_best.tar
