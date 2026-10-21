import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from PIL import Image
import timm
from timm.data import resolve_model_data_config,create_transform

BACKBONES={'resnet50':'resnet50','efficientnetv2':'tf_efficientnetv2_s','convnext-tiny':'convnext_tiny'}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

class FoodDataset(Dataset):
    def __init__(self,rows,labels,transform,regression=False): self.rows=rows;self.labels=labels;self.transform=transform;self.regression=regression
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image']) as raw: image=raw.convert('RGB')
        if 'bbox' in r: image=image.crop(r['bbox'])
        target=float(r['weight_g']) if self.regression else self.labels.index(r['class_name'])
        return self.transform(image),torch.tensor(target,dtype=torch.float32 if self.regression else torch.long)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--backbone',choices=BACKBONES,required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true');p.add_argument('--regression',action='store_true');a=p.parse_args();seed_all(a.seed)
    rows=[json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s];labels=sorted(set(r['class_name'] for r in rows));device='cuda' if torch.cuda.is_available() else 'cpu'
    if a.regression and any('weight_g' not in r for r in rows): raise ValueError('Regression requires measured weight_g per sample; no pseudo item weights')
    model=timm.create_model(BACKBONES[a.backbone],pretrained=True,num_classes=1 if a.regression else len(labels)).to(device)
    cfg=resolve_model_data_config(model)
    train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='validation']
    if not train_rows or not val_rows: raise ValueError('Train and validation splits must be nonempty')
    train=FoodDataset(train_rows,labels,create_transform(**cfg,is_training=True),a.regression);val=FoodDataset(val_rows,labels,create_transform(**cfg,is_training=False),a.regression)
    counts={label:sum(r['class_name']==label for r in train_rows) for label in labels}
    sampler=WeightedRandomSampler([1/counts[r['class_name']] for r in train_rows],len(train_rows),generator=torch.Generator().manual_seed(a.seed)) if not a.regression else None
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,shuffle=sampler is None,num_workers=2)
    validation=DataLoader(val,batch_size=a.batch_size,num_workers=2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda');out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=0;best=float('inf')
    if a.resume and (out/'last.pt').exists():
        checkpoint=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if checkpoint['labels']!=labels or checkpoint['backbone']!=BACKBONES[a.backbone]: raise ValueError('Resume taxonomy/backbone mismatch')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scaler.load_state_dict(checkpoint['scaler']);start=checkpoint['epoch']+1;best=checkpoint['best'];torch.set_rng_state(checkpoint['torch_rng'].cpu());np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        if device=='cuda' and checkpoint.get('cuda_rng'): torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        if sampler and checkpoint.get('sampler_rng') is not None: sampler.generator.set_state(checkpoint['sampler_rng'])
    criterion=torch.nn.L1Loss() if a.regression else torch.nn.CrossEntropyLoss()
    for epoch in range(start,a.epochs):
        model.train()
        for images,target in loader:
            images,target=images.to(device),target.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,enabled=device=='cuda'):
                prediction=model(images);prediction=prediction[:,0] if a.regression else prediction
                loss=criterion(prediction,target)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        model.eval();error=0
        with torch.inference_mode():
            for images,target in validation:
                prediction=model(images.to(device));target=target.to(device)
                error+=float((prediction[:,0]-target).abs().sum()) if a.regression else float((prediction.argmax(1)!=target).sum())
        score=error/len(val);improved=score<best;best=min(best,score)
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'labels':labels,'backbone':BACKBONES[a.backbone],'regression':a.regression,'seed':a.seed,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,'sampler_rng':sampler.generator.get_state() if sampler else None}
        torch.save(checkpoint,out/'last.tmp');os.replace(out/'last.tmp',out/'last.pt')
        if improved: torch.save({k:checkpoint[k] for k in ['model','labels','backbone','regression','epoch','best','seed']},out/'best.pt')
        print(json.dumps({'epoch':epoch,'validation_mae_g' if a.regression else 'validation_error_rate':score,'best':best}),flush=True)
if __name__=='__main__': main()
