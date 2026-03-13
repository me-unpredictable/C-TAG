# # # this script is to run all training experiments in terminal one by one

# # # th 0.2 gcnn 1 tcnn 2 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_quad_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name quad --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_nexus_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name nexus --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_little_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name little --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_hyang_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name hyang --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_gates_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name gates --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_deathCircle_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name deathCircle --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_coupa_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name coupa --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.2_bookstore_rt --thres 0.2 --n_gcnn 1 --n_tcnn 2 --scene_name bookstore --num_epochs 250 --lr 0.01

# # # th 0.4 gcnn 2 tcnn 2 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore
# python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_quad_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name quad --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_nexus_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name nexus --num_epochs 250 --lr 0.01
# python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_little_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name little --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_hyang_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name hyang --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_gates_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name gates --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_deathCircle_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name deathCircle --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_coupa_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name coupa --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.4_bookstore_rt --thres 0.4 --n_gcnn 1 --n_tcnn 2 --scene_name bookstore --num_epochs 250 --lr 0.01
# # # we learend that increasing GRaph layers affect results and results start degrading


# # # th 0.6 gcnn 1 tcnn 2 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_quad_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name quad --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_nexus_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name nexus --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_little_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name little --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_hyang_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name hyang --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_gates_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name gates --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_deathCircle_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name deathCircle --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_coupa_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name coupa --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.6_bookstore_rt --thres 0.6 --n_gcnn 1 --n_tcnn 2 --scene_name bookstore --num_epochs 250 --lr 0.01

# # # th 0.8 gcnn 1 tcnn 2 quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore 
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_quad_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name quad --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_nexus_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name nexus --num_epochs 250 --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_little_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name little --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_hyang_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name hyang --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_gates_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name gates --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_deathCircle_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name deathCircle --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_coupa_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name coupa --num_epochs 250  --lr 0.01
python train.py --dataset SDD --dataset_path ../datasets/SDD/archive --tag CTAG.g1.t2.th.0.8_bookstore_rt --thres 0.8 --n_gcnn 1 --n_tcnn 2 --scene_name bookstore --num_epochs 250 --lr 0.01


# # echo "All training experiments completed."
