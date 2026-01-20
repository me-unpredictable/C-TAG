from ast import arg
from operator import le
from re import A, split
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse


argparse = argparse.ArgumentParser()
argparse.add_argument('--dataset_root', type=str, default='/path/to/dataset/root', help='Path to the dataset root directory')
argparse.add_argument('--dataset_path', type=str, default='./processed', help='Path to save processed data')
argparse.add_argument('--dataset_name', type=str, default='SDD', help='Name of the dataset being used')
args = argparse.parse_args()
dataset_root = args.dataset_root
dataset_path = args.dataset_path
dataset_name = args.dataset_name



# ==========================================
# 0. VISUAL BACKBONE (From Step 1 - Approved)
# ==========================================
class VisualBackbone(nn.Module):
    def __init__(self):
        super(VisualBackbone, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)
        
        # Keep layers up to Layer 4 (output stride 32)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        
        # Freeze early layers
        for name, child in self.backbone.named_children():
            if name in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']:
                for param in child.parameters():
                    param.requires_grad = False
            
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# 1.CONTEXT EXTRACTOR (The Fusion Logic)
# ==========================================
def extract_features_for_agent(feature_map, agent_df, orig_w, orig_h):
    """
    Extracts visual features for a specific agent's trajectory.
    """
    # Get coordinates and normalize to [-1, 1]
    # Note: We use original image dimensions because the relative position 
    # of the agent (e.g., "center of image") is invariant to resizing.
    
    # cx / orig_w gives 0.0 to 1.0. 
    # Multiply by 2 -> 0.0 to 2.0. 
    # Subtract 1 -> -1.0 to 1.0.
    norm_x = 2 * (agent_df['cx'].values / orig_w) - 1
    norm_y = 2 * (agent_df['cy'].values / orig_h) - 1
    
    # Prepare grid for grid_sample: [1, Time, 1, 2]
    # We treat the trajectory as a "line of pixels" we want to sample
    grid = torch.tensor(np.stack([norm_x, norm_y], axis=1), dtype=torch.float32)
    grid = grid.unsqueeze(0).unsqueeze(2) 
    
    # Move to same device as feature map
    grid = grid.to(feature_map.device)
    
    # Sample
    # Feature map shape: [1, 2048, H_feat, W_feat]
    # Output shape: [1, 2048, Time, 1]
    sampled = F.grid_sample(feature_map, grid, align_corners=False)
    
    # Reshape to [Time, Channels] (Batch size 1 assumed for single scene)
    return sampled.squeeze(0).squeeze(-1).permute(1, 0)

def extract_map_features(img_path,map_file_name,dataset_path='./processed'):
    # now we are all set for processing the dataset

    resize_dim = (512, 512) # standard input size for ResNet50

    # # --- B. LOAD IMAGE ---
    orig_img = Image.open(img_path).convert('RGB')
    
    orig_w, orig_h = orig_img.size
    print(f"Original Image Size: {orig_w} x {orig_h}")

    # Prepare input tensor
    transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(orig_img).unsqueeze(0) # [1, 3, 512, 512]

    # --- C. RUN VISUAL BACKBONE ---
    model = VisualBackbone()
    model.eval() # Eval mode for Batch Norm stability
    
    with torch.no_grad():
        feature_map = model(input_tensor)
    
    print(f"Feature Map Shape: {feature_map.shape}") # Should be [1, 2048, 16, 16]

    
    # save the processed map features to processed directory
    save_path=os.path.join(dataset_path,map_file_name)
    torch.save(feature_map, save_path)


def save_map_features(dataset_root,dataset_path='./processed'):
    
    # check if processed dir exists
    # we will store all of our map and trajectory data in it.
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    # check the dataset root
    dirs=os.listdir(dataset_root)

    if dataset_name=='SDD':
        # dataset root must contain annotation and videos folders for SDD
        assert 'annotations' in dirs, "annotations folder not found in dataset root"
        # only checking for annotation as we just need map and trajectory data
        # get avilable scenes
        scenes=os.listdir(os.path.join(dataset_root,'annotations'))
        # check if maps dir exists in processed dir
        if not os.path.exists(os.path.join(dataset_path,'maps')):
            os.makedirs(os.path.join(dataset_path,'maps'))

        # now we will start processing each split
        for s in scenes:
            # enlist number of videos in the scene
            videos=os.listdir(os.path.join(dataset_root,'annotations',s))
            for v in videos:
                # name of the processed map file
                # s_v_map.pt in processed/train or val or test dir
                map_file_name=os.path.join('maps','{}_{}_map.pt'.format(s,v))
                # enlist all jpg files in the video folder
                img_files=os.listdir(os.path.join(dataset_root,'annotations',s,v))
                # filter only jpg files
                img_files=[f for f in img_files if f.endswith('.jpg')]
                # if there are more than 1 jpg files then we will process reference.jpg
                # otherwise whatever named jpg file is there we will process it
                if len(img_files)>1:
                    img_path=os.path.join(dataset_root,'annotations',s,v,'reference.jpg')
                else:
                    try:
                        img_path=os.path.join(dataset_root,'annotations',s,v,img_files[0])
                    except IndexError:
                        print(f"No jpg files found in {os.path.join(dataset_root,'annotations',s,v)}")
                        continue
                # extract map features
                extract_map_features(img_path,map_file_name)
    print("All map features extracted and saved.")
print("Map feature extraction module loaded.")
save_map_features(dataset_root,dataset_path)
