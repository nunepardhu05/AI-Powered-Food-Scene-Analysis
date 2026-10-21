"""Evaluate measured, paired predictions. Does not train or invent metrics."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def metrics(rows):
    y=np.array([r['weight_g'] for r in rows],float);pred=np.array([r['predicted_g'] for r in rows],float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all() or (y<0).any(): raise ValueError('Finite nonnegative measured weights required')
    nonzero=y>0
    return {'mae_g':float(mean_absolute_error(y,pred)),'rmse_g':float(math.sqrt(mean_squared_error(y,pred))),'mape_percent':float(np.mean(abs((pred[nonzero]-y[nonzero])/y[nonzero]))*100) if nonzero.any() else None,'mape_excluded_zero_count':int((~nonzero).sum()),'r2':float(r2_score(y,pred)) if len(y)>1 and np.var(y)>0 else None,'count':len(y), 'calorie_mae_kcal':float(np.mean([abs(r['kcal']-r['predicted_kcal']) for r in rows])) if all('kcal' in r and 'predicted_kcal' in r for r in rows) else None,'carb_mae_g':float(np.mean([abs(r['carbs_g']-r['predicted_carbs_g']) for r in rows])) if all('carbs_g' in r and 'predicted_carbs_g' in r for r in rows) else None,'interval_coverage':float(np.mean([r['low_g']<=r['weight_g']<=r['high_g'] for r in rows])) if all('low_g' in r and 'high_g' in r for r in rows) else None,'latency_ms_median':float(np.median([r['latency_ms'] for r in rows])) if all('latency_ms' in r for r in rows) else None,'latency_ms_p95':float(np.percentile([r['latency_ms'] for r in rows],95)) if all('latency_ms' in r for r in rows) else None}

def detection_metrics(gt,pred):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth=COCO(gt);predictions=truth.loadRes(pred);evaluator=COCOeval(truth,predictions,'bbox');evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    return {'map50':float(evaluator.stats[1]),'map50_95':float(evaluator.stats[0])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);p.add_argument('--coco-gt');p.add_argument('--coco-predictions');p.add_argument('--model-files',nargs='*',default=[]);a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.predictions).read_text().splitlines() if s]
    if any(r['split']!='test' for r in rows): raise ValueError('Final reporting requires untouched test split')
    report={'overall':metrics(rows),'per_class':{c:metrics([r for r in rows if r['class_name']==c]) for c in sorted(set(r['class_name'] for r in rows))},'model_size_mb':sum(Path(f).stat().st_size for f in a.model_files)/1e6 if a.model_files else None}
    if a.coco_gt and a.coco_predictions: report.update(detection_metrics(a.coco_gt,a.coco_predictions))
    Path(a.out).write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
