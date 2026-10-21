"""Apply selected model and calibration to the shared held-out feature rows."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--selection-dir',required=True);p.add_argument('--manifest',required=True);p.add_argument('--cnn-predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.selection_dir);selection=json.loads((root/'selection.json').read_text());data=np.load(a.features,allow_pickle=False)
    records={str(r['sample_id']):r for r in [json.loads(s) for s in Path(a.manifest).read_text().splitlines() if s]}
    if selection['winner']=='xgboost':
        artifact=joblib.load(root/'xgboost.joblib');features=np.column_stack([data['features'],artifact['pca'].transform(data['embeddings'])]);predictions=artifact['model'].predict(features)
    else:
        cnn={str(r['sample_id']):r['predicted_g'] for r in [json.loads(s) for s in Path(a.cnn_predictions).read_text().splitlines() if s]}
        predictions=np.array([cnn[str(s)] if split!='train' else np.nan for s,split in zip(data['sample_id'],data['split'])])
    radius=selection['conformal_radius_g']
    with Path(a.output).open('w') as out:
        for i,sid in enumerate(data['sample_id']):
            if data['split'][i]!='test':continue
            record=records[str(sid)];prediction=max(0,float(predictions[i]))
            out.write(json.dumps({**record,'predicted_g':prediction,'low_g':max(0,prediction-radius),'high_g':prediction+radius,'selected_model':selection['winner']})+'\n')
if __name__=='__main__':main()
