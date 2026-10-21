import argparse
import json
import shutil
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s];labels=sorted(set(r['class_name'] for r in rows));categories=[{'id':i,'name':label,'supercategory':'food'} for i,label in enumerate(labels)]
    for split,folder in [('train','train'),('validation','valid'),('test','test')]:
        target=Path(a.out)/folder;target.mkdir(parents=True,exist_ok=True);images=[];annotations=[];seen={}
        for row in rows:
            if row['split']!=split:continue
            if 'bbox' not in row: raise ValueError('Detection export requires human-annotated bbox [x1,y1,x2,y2]')
            path=Path(row['image']);key=str(path.resolve())
            if key not in seen:
                i=len(images);seen[key]=i;name=f'{row["sha256"]}{path.suffix.lower()}'
                with Image.open(path) as image:w,h=image.size
                images.append({'id':i,'file_name':name,'width':w,'height':h});shutil.copy2(path,target/name)
            x1,y1,x2,y2=row['bbox']
            if x2<=x1 or y2<=y1:raise ValueError('Invalid bbox')
            annotations.append({'id':len(annotations),'image_id':seen[key],'category_id':labels.index(row['class_name']),'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
        (target/'_annotations.coco.json').write_text(json.dumps({'images':images,'annotations':annotations,'categories':categories}))
    (Path(a.out)/'labels.json').write_text(json.dumps(labels))
if __name__=='__main__':main()
