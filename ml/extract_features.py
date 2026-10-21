"""Measured-geometry feature extraction. Never derives false metric depth from relative depth."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import timm
from PIL import Image
from timm.data import resolve_model_data_config,create_transform
from app.volume import estimate_volume
from app.nutrition import foods

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s]
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=timm.create_model(ckpt['backbone'],pretrained=False,num_classes=len(ckpt['labels']));model.load_state_dict(ckpt['model']);model.reset_classifier(0);model.eval();transform=create_transform(**resolve_model_data_config(model),is_training=False)
    features=[];embeddings=[];weights=[];ids=[];splits=[]
    with torch.inference_mode():
        for row in rows:
            required={'geometry_npz','weight_g','food_id','sample_id'}
            if not required.issubset(row): raise ValueError(f'Measured weight/geometry metadata required: {required}')
            g=np.load(row['geometry_npz'],allow_pickle=False);mask=g['mask'].astype(bool);depth=g['depth_cm'];plate=g['plate_mask'].astype(bool)
            volume=estimate_volume(depth,mask,plate,float(g['cm_per_pixel']),calibrated=bool(g['calibrated']))['volume_ml']
            yy,xx=np.where(mask)
            if not len(xx): raise ValueError('Empty food mask')
            food=foods()[row['food_id']];physics=volume*food['density']*food['packing']
            features.append([volume,mask.sum(),xx.max()-xx.min()+1,yy.max()-yy.min()+1,mask.mean(),depth[mask].mean(),depth[mask].std(),ckpt['labels'].index(row['class_name']),physics])
            with Image.open(row['image']) as raw: image=raw.convert('RGB')
            if 'bbox' in row:image=image.crop(row['bbox'])
            embeddings.append(model(transform(image).unsqueeze(0))[0].numpy());weights.append(row['weight_g']);ids.append(str(row['sample_id']));splits.append(row['split'])
    np.savez_compressed(a.output,features=np.array(features),embeddings=np.array(embeddings),weight_g=np.array(weights),sample_id=np.array(ids),split=np.array(splits))
if __name__=='__main__':main()
