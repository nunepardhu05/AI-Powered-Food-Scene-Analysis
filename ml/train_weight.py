"""Fit PCA only on train, choose model on validation, calibrate on held-out groups."""
import argparse
import json
import math
from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

FEATURES=['volume_ml','mask_area','bbox_width','bbox_height','relative_area','depth_mean','depth_std','class_id','physics_weight_g']

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--cnn-predictions',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    data=np.load(a.features,allow_pickle=False)
    required=['features','embeddings','weight_g','split','sample_id']
    if any(k not in data for k in required): raise ValueError(f'NPZ needs {required}')
    X=data['features'];E=data['embeddings'];y=data['weight_g'];splits=data['split'];ids=data['sample_id']
    if X.shape[1]!=len(FEATURES) or len(set(ids))!=len(ids): raise ValueError('Invalid feature shape or duplicate sample IDs')
    train=splits=='train';val=splits=='validation';cal=splits=='calibration'
    if min(train.sum(),val.sum(),cal.sum())<2: raise ValueError('Need train, validation and independent calibration rows')
    if not np.isfinite(X).all() or not np.isfinite(E).all() or not np.isfinite(y).all(): raise ValueError('Features and labels must be finite')
    pca=PCA(n_components=min(32,E.shape[1],int(train.sum())-1),random_state=a.seed).fit(E[train])
    features=np.column_stack([X,pca.transform(E)])
    candidates=[]
    for depth in [3,5,7]:
        model=XGBRegressor(n_estimators=500,max_depth=depth,learning_rate=.04,subsample=.8,colsample_bytree=.8,objective='reg:absoluteerror',tree_method='hist',random_state=a.seed,n_jobs=2)
        model.fit(features[train],y[train]);score=mean_absolute_error(y[val],model.predict(features[val]));candidates.append((score,model))
    xgb_score,model=min(candidates,key=lambda x:x[0])
    cnn_rows=[json.loads(s) for s in Path(a.cnn_predictions).read_text().splitlines() if s];cnn={str(r['sample_id']):r for r in cnn_rows}
    for i in np.where(val|cal)[0]:
        row=cnn.get(str(ids[i]))
        if not row or row['split']!=splits[i] or abs(row['weight_g']-float(y[i]))>1e-5: raise ValueError('CNN predictions must match the same validation/calibration IDs and labels')
    cnn_score=mean_absolute_error(y[val],[cnn[str(s)]['predicted_g'] for s in ids[val]])
    winner='xgboost' if xgb_score<=cnn_score else 'cnn'
    calibration=model.predict(features[cal]) if winner=='xgboost' else np.array([cnn[str(s)]['predicted_g'] for s in ids[cal]])
    residual=np.abs(y[cal]-calibration);rank=math.ceil((len(residual)+1)*.9)
    if rank>len(residual): raise ValueError('At least 9 calibration samples needed for finite 90% split-conformal intervals')
    radius=float(np.sort(residual)[rank-1])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'pca':pca,'features':FEATURES},out/'xgboost.joblib')
    report={'winner':winner,'validation_mae_g':{'xgboost':float(xgb_score),'cnn':float(cnn_score)},'conformal_radius_g':radius,'nominal_coverage':.9,'calibration_count':len(residual),'seed':a.seed,'limitations':'Marginal split-conformal coverage assumes exchangeable data; not guaranteed per food class or after distribution shift'}
    (out/'selection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
