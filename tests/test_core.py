import copy
import io
import math
import numpy as np
import pytest
from PIL import Image
from app.schemas import Profile,Target,MealRequest
from app.nutrition import nutrients,aggregate,split_nutrients,household,foods,decompose
from app.rules_engine import guidance,alerts,bmi,energy,validate_rule,catalog,blocked_food
from app.volume import estimate_volume,resolve_overlaps,physics_weight
from app.planner import plan
from app.vision import analyze,load_image


def test_nutrients_and_ranges():
    n=nutrients('rice',200,.1)
    assert n['kcal']=={'value':260.0,'low':234.0,'high':286.0}
    assert n['carbs_g']['value']==56.4
    assert n['purines_mg'] is None


def test_missing_not_zero():
    result=aggregate([nutrients('rice',100),nutrients('apple',100)])
    assert result['purines_mg'] is None
    assert result['kcal']['value']==182


def test_portion_percentages():
    total=nutrients('rice',200)
    portions=split_nutrients(total,[25,75])
    assert portions[0]['kcal']['value']==65
    assert portions[1]['carbs_g']['value']==42.3
    with pytest.raises(ValueError):split_nutrients(total,[10,10])


def test_unit_conversion():
    h=household(240,1,1)
    assert h['cups_240ml']==1
    assert h['katori_150ml']==1.6
    assert physics_weight(100,.8,.5)['value']==40


def test_bmi_energy():
    p=Profile(age=30,sex='male',height_cm=180,weight_kg=80,activity='sedentary')
    assert bmi(p)==24.7
    assert energy(p)==2136
    assert energy(Profile(sex='unspecified')) is None


@pytest.mark.parametrize('condition',['phenylketonuria','maple-syrup-urine-disease','eating-disorder-history','anorexia-nervosa','organ-transplant','icu-recovery','tube-feeding'])
def test_tier3_no_restriction(condition):
    p=Profile(conditions=[condition],goal='lose',targets={'kcal':Target(maximum=1500,unit='kcal')})
    g=guidance(p)
    assert g['tier']==3 and g['targets']=={} and g['bmi'] is None
    assert plan(p)['meals']==[]
    assert alerts(nutrients('rice',500),p)==[]


@pytest.mark.parametrize('condition',['ckd-stage-4','hemodialysis','heart-failure','cirrhosis','cancer-treatment','pregnancy','crohn-disease','bariatric-sleeve-surgery','epilepsy','dysphagia'])
def test_supervised(condition):
    p=Profile(conditions=[condition]);g=guidance(p)
    assert g['tier']>=2 and g['banners']
    assert not plan(p)['meals']


def test_pediatric_pregnancy():
    assert guidance(Profile(age=12))['tier']==2
    assert guidance(Profile(pregnancy_trimester=1))['tier']==2


def test_unlisted_fallback():
    g=guidance(Profile(conditions=['unlisted medical condition']))
    assert g['unmatched']==['unlisted medical condition']
    assert not g['can_plan']
    assert g['targets']=={}


def test_free_text_high_risk():
    assert guidance(Profile(conditions=['history of severe eating disorder']))['tier']==3
    assert guidance(Profile(conditions=['receiving cancer chemotherapy']))['tier']==2


def test_clinician_override():
    p=Profile(conditions=['hypertension'],targets={'sodium_mg':Target(maximum=2500,unit='mg')})
    g=guidance(p)
    assert g['targets']['sodium_mg']['maximum']==2500
    assert any('overrides' in c for c in g['conflicts'])


def test_stricter_conflict(monkeypatch):
    from app import rules_engine as r
    rule=copy.deepcopy(catalog()[0]);rule.update(id='a',name='A',tier=1,limits={'sodium_mg':{'maximum':2300,'unit':'mg'}})
    other=copy.deepcopy(rule);other.update(id='b',name='B',limits={'sodium_mg':{'maximum':1500,'unit':'mg'}})
    monkeypatch.setattr(r,'catalog',lambda:[rule,other])
    g=r.guidance(Profile(conditions=['a','b']))
    assert g['targets']['sodium_mg']['maximum']==1500
    assert any('stricter' in c for c in g['conflicts'])


def test_incompatible_limits_block_plan(monkeypatch):
    from app import rules_engine as r
    rule=copy.deepcopy(catalog()[0]);rule.update(id='a',name='A',tier=1,limits={'sodium_mg':{'minimum':2000,'unit':'mg'}})
    other=copy.deepcopy(rule);other.update(id='b',name='B',limits={'sodium_mg':{'maximum':1500,'unit':'mg'}})
    monkeypatch.setattr(r,'catalog',lambda:[rule,other])
    assert plan(Profile(conditions=['a','b']))['status']=='infeasible'


@pytest.mark.parametrize('condition,food',[('celiac-disease','chapati'),('lactose-intolerance','yogurt')])
def test_avoid_foods(condition,food):
    p=Profile(conditions=[condition])
    assert blocked_food(foods()[food],p)
    assert alerts(nutrients(food,100),p,[food])[0]['level']=='danger'


def test_allergen_tier3():
    p=Profile(conditions=['phenylketonuria'],allergies=['peanut'])
    result=alerts(nutrients('peanut',10),p,['peanut'])
    assert result[0]['level']=='danger'


def test_plan_allergies_and_limits():
    p=Profile(sex='female',allergies=['milk','peanut','soy'],conditions=['hypertension'],targets={'kcal':Target(maximum=1800,unit='kcal'),'sodium_mg':Target(maximum=1800,unit='mg')})
    result=plan(p)
    assert result['status']=='illustrative'
    assert len(result['meals'])==5
    assert result['totals']['kcal']['high']<=1800.1
    assert result['totals']['sodium_mg']['high']<=1800.1
    assert all(i['food_id'] not in ('yogurt','peanut','tofu') for m in result['meals'] for i in m['items'])


def test_infeasible_plan():
    result=plan(Profile(sex='female',targets={'sodium_mg':Target(maximum=0,unit='mg')}))
    assert result['status']=='infeasible' and result['meals']==[]


def test_unknown_allergy_blocks():
    assert plan(Profile(allergies=['rare ingredient'],sex='female'))['status']=='infeasible'


def test_validator():
    rules=catalog()
    assert len(rules)>130
    assert all(validate_rule(r) for r in rules)
    rule=copy.deepcopy(rules[0]);rule['status']='reviewed'
    with pytest.raises(ValueError):validate_rule(rule)
    rule=copy.deepcopy(rules[0]);rule['limits']={'sodium_mg':{'maximum':100,'unit':'g'}}
    with pytest.raises(ValueError):validate_rule(rule)


def test_volume_hemisphere():
    radius=5;scale=.025
    axis=np.arange(-7,7,scale);x,y=np.meshgrid(axis,axis);r2=x*x+y*y;mask=r2<=radius*radius
    height=np.sqrt(np.maximum(0,radius*radius-r2));plane=100+.01*x+.02*y
    result=estimate_volume(plane-height,mask,np.ones_like(mask),scale,calibrated=True)
    assert result['volume_ml']==pytest.approx(2/3*math.pi*radius**3,rel=.002)


def test_uncalibrated_depth_rejected():
    with pytest.raises(ValueError,match='Calibrate'):estimate_volume(np.ones((10,10)),np.ones((10,10)),np.ones((10,10)),1)


def test_overlap_priority():
    masks=np.ones((2,10,10),bool)
    out,flags=resolve_overlaps(masks,[.2,.9])
    assert out[0].sum()==0 and out[1].sum()==100 and flags==[True,True]


def test_mixed_recipe_is_flagged():
    result=decompose('chicken rice bowl',200)
    assert result['estimated'] and result['assumptions']
    assert sum(x['grams'] for x in result['items'])==200


@pytest.mark.parametrize('grams',[0,-1,5001,float('nan'),float('inf')])
def test_invalid_portions(grams):
    from app.schemas import FoodEntry
    with pytest.raises(ValueError): FoodEntry(food_id='rice',grams=grams)


def test_small_large_valid_portions():
    assert nutrients('rice',1)['kcal']['value']==1.3
    assert nutrients('rice',5000)['kcal']['value']==6500


def test_dark_unknown_missing_reference():
    result=analyze(Image.new('RGB',(100,100),(0,0,0)))
    assert result['status']=='models_unavailable'
    assert result['items']==[]
    assert any('Dark' in w for w in result['warnings'])
    assert any('Missing reference' in w for w in result['warnings'])


def test_invalid_image():
    with pytest.raises(ValueError):load_image(b'not an image')


def test_medication_alerts():
    p=Profile(medications=['warfarin','levothyroxine','metformin'])
    result=alerts(nutrients('rice',100),p)
    assert len(result)==3 and all('Ask your doctor' in x['message'] for x in result)


def test_plan_swaps_satisfy_same_day():
    from app.planner import respects
    p=Profile(sex='female',conditions=['hypertension'],targets={'kcal':Target(maximum=1800,unit='kcal')})
    result=plan(p)
    assert result['status']=='illustrative'
    assert len(set(m['name'] for m in result['meals']))>3
    assert any(m['swaps'] for m in result['meals'])
    targets=guidance(p)['targets']
    for i,m in enumerate(result['meals']):
        for swap in m['swaps']:
            revised=aggregate([x['nutrients'] if j!=i else swap['nutrients'] for j,x in enumerate(result['meals'])])
            assert respects(revised,targets)


def test_clinician_meal_carb_target():
    p=Profile(meal_targets={'carbs_g':Target(maximum=30,unit='g')})
    result=alerts(nutrients('rice',200),p)
    assert any('per-meal limit 30' in a['message'] for a in result)
    restricted=Profile(conditions=['phenylketonuria'],meal_targets={'carbs_g':Target(maximum=30,unit='g')})
    assert guidance(restricted)['meal_targets']=={}
    assert alerts(nutrients('rice',200),restricted)==[]
