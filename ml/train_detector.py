import argparse
from pathlib import Path
from ml.train_classifier import seed_all

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--size',choices=['small','base'],required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--resume');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    from rfdetr import RFDETRSmall,RFDETRBase
    seed_all(a.seed)
    for split in ['train','valid','test']:
        if not (Path(a.data)/split/'_annotations.coco.json').exists(): raise ValueError('RF-DETR needs train/valid/test COCO exports from the shared manifest')
    model=(RFDETRSmall if a.size=='small' else RFDETRBase)()
    options={'dataset_dir':a.data,'epochs':a.epochs,'batch_size':1,'grad_accum_steps':8,'lr':1e-4,'output_dir':a.out,'seed':a.seed,'device':'cuda','run_test':False,'tensorboard':False}
    if a.resume: options['resume']=a.resume
    model.train(**options)
if __name__=='__main__':main()
