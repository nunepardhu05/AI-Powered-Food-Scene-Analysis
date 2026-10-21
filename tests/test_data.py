from pathlib import Path
import pytest
from ml.data import prepare,stable_split
from ml.evaluate import metrics

def test_group_split_duplicate_and_conflict(tmp_path):
    a=tmp_path/'a.bin';b=tmp_path/'b.bin';a.write_bytes(b'same');b.write_bytes(b'same')
    taxonomy={'sample:rice':{'name':'rice','fdc_id':None},'sample:other':{'name':'other','fdc_id':None}}
    records=[{'dataset':'sample','image':str(a),'dish_id':'dish1','label':'rice'},{'dataset':'sample','image':str(b),'dish_id':'dish2','label':'rice'}]
    result=prepare(records,taxonomy)
    assert len(result)==1
    assert result[0]['split']==stable_split(result[0]['group'])
    records[1]['label']='other'
    with pytest.raises(ValueError,match='conflicting'):prepare(records,taxonomy)


def test_metrics_and_missing_metrics():
    result=metrics([{'weight_g':100,'predicted_g':110},{'weight_g':200,'predicted_g':180}])
    assert result['mae_g']==15
    assert result['rmse_g']==pytest.approx(15.811388)
    assert result['mape_percent']==10
    assert result['calorie_mae_kcal'] is None
    assert result['latency_ms_median'] is None
