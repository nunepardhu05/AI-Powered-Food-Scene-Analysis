import argparse
import json
import time
from pathlib import Path
import torch
import timm
from timm.data import resolve_model_data_config,create_transform
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if not checkpoint['regression']: raise ValueError('A trained CNN weight-regression checkpoint is required')
    model=timm.create_model(checkpoint['backbone'],pretrained=False,num_classes=1);model.load_state_dict(checkpoint['model']);model.eval();transform=create_transform(**resolve_model_data_config(model),is_training=False)
    rows=[json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s]
    with Path(a.output).open('w') as out,torch.inference_mode():
        for r in rows:
            if r['split']=='train': continue
            with Image.open(r['image']) as raw: image=raw.convert('RGB')
            if 'bbox' in r:image=image.crop(r['bbox'])
            tensor=transform(image).unsqueeze(0);start=time.perf_counter();weight=max(0,float(model(tensor)[0,0]));elapsed=(time.perf_counter()-start)*1000
            out.write(json.dumps({**r,'predicted_g':weight,'latency_ms':elapsed})+'\n')
if __name__=='__main__':main()
