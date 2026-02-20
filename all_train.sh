# this script is to run all training experiments in terminal one by one

# th 0.4 gcnn 1 tcnn 3 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_quad --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name quad --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_nexus --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name nexus --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_little --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name little --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_hyang --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name hyang --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_gates --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name gates --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_deathCircle --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name deathCircle --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_coupa --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name coupa --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.4_bookstore --thres 0.4 --n_gcnn 1 --n_tcnn 3 --scene_name bookstore --num_epochs 250 --lr 0.01

# th 0.4 gcnn 2 tcnn 3 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_quad --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name quad --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_nexus --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name nexus --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_little --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name little --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_hyang --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name hyang --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_gates --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name gates --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_deathCircle --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name deathCircle --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_coupa --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name coupa --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g2.t3.th.0.4_bookstore --thres 0.4 --n_gcnn 2 --n_tcnn 3 --scene_name bookstore --num_epochs 250 --lr 0.01
# we learend that increasing GRaph layers affect results and results start degrading


# th 0.6 gcnn 1 tcnn 3 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_quad --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name quad --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_nexus --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name nexus --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_little --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name little --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_hyang --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name hyang --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_gates --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name gates --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_deathCircle --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name deathCircle --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_coupa --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name coupa --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.6_bookstore --thres 0.6 --n_gcnn 1 --n_tcnn 3 --scene_name bookstore --num_epochs 250 --lr 0.01

# th 0.8 gcnn 1 tcnn 3 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_quad --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name quad --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_nexus --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name nexus --num_epochs 250 --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_little --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name little --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_hyang --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name hyang --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_gates --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name gates --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_deathCircle --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name deathCircle --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_coupa --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name coupa --num_epochs 250  --lr 0.01
#python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t3.th.0.8_bookstore --thres 0.8 --n_gcnn 1 --n_tcnn 3 --scene_name bookstore --num_epochs 250 --lr 0.01


echo "All training experiments completed."
