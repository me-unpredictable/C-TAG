# this script is to run all training experiments in terminal one by one

# th 0.2 gcnn 2 tcnn 10 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_quad --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name quad --num_epochs 150  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_nexus --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name nexus --num_epochs 150 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_little --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name little --num_epochs 150  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_hyang --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name hyang --num_epochs 150  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_gates --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name gates --num_epochs 150  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_deathCircle --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name deathCircle --num_epochs 150  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_coupa --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name coupa --num_epochs 150  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t10.th.0.2_bookstore --thres 0.2 --n_gcnn 1 --n_tcnn 10 --scene_name bookstore --num_epochs 150 --lr 0.01

# th 0.2 gcnn 3 tcnn 10 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_quad --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name quad --num_epochs 150  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_nexus --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name nexus --num_epochs 150 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_little --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name little --num_epochs 150 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_hyang --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name hyang --num_epochs 150 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_gates --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name gates --num_epochs 150 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_deathCircle --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name deathCircle --num_epochs 150 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_coupa --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name coupa --num_epochs 150 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t10.th.0.2_bookstore --thres 0.2 --n_gcnn 2 --n_tcnn 10 --scene_name bookstore --num_epochs 150 --lr 0.01

echo "All training experiments completed."