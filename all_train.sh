# this script is to run all training experiments in terminal one by one

# th 0.4 gcnn 2 tcnn 3 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_quad --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name quad --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_nexus --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name nexus --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_little --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name little --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_hyang --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name hyang --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_gates --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name gates --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_deathCircle --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name deathCircle --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_coupa --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name coupa --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_bookstore --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name bookstore --num_epochs 250 --lr 0.01

# th 0.4 gcnn 3 tcnn 3 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_quad --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name quad --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_nexus --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name nexus --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_little --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name little --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_hyang --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name hyang --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_gates --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name gates --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_deathCircle --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name deathCircle --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_coupa --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name coupa --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_bookstore --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name bookstore --num_epochs 250 --lr 0.01

echo "All training experiments completed."