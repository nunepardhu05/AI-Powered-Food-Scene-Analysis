from pathlib import Path
import re
import difflib
import yaml
from .schemas import Profile
from .nutrition import foods, NUTRIENTS
ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = 'Supports awareness and dietary management. Not medical advice; does not diagnose, treat or cure. Clinician targets take priority. Draft rules require qualified clinical review before patient use.'
UNITS = {k: ('kcal' if k=='kcal' else k.rsplit('_',1)[-1]) for k in NUTRIENTS}
UNITS['fluid_ml'] = 'ml'

def validate_rule(rule):
    required = {'id','name','group','version','tier','status','personalized','limits','per_meal_limits','prefer','avoid','source','last_reviewed','reviewer','icd10cm','icd11'}
    missing = required-set(rule)
    if missing: raise ValueError(f'Missing fields: {sorted(missing)}')
    if rule['tier'] not in (1,2,3) or rule['status'] not in ('draft','reviewed'): raise ValueError('Invalid tier or status')
    if not re.fullmatch(r'[a-z0-9-]+',rule['id']): raise ValueError('Invalid rule ID')
    if rule['status']=='reviewed' and (not rule['reviewer'] or not rule['last_reviewed']): raise ValueError('Reviewed rules require date and reviewer')
    if not isinstance(rule['prefer'],list) or not isinstance(rule['avoid'],list): raise ValueError('Food guidance must be lists')
    if not rule['source'].get('url','').startswith('https://'): raise ValueError('HTTPS guideline URL required')
    from .schemas import Target
    for scope in ['limits','per_meal_limits']:
        for nutrient, target in rule[scope].items():
            if nutrient not in UNITS or target.get('unit')!=UNITS[nutrient]: raise ValueError(f'Invalid nutrient or unit: {nutrient}')
            Target.model_validate(target)
    return True

def catalog():
    rules=[]
    for path in sorted((ROOT/'app/rules').glob('*.yaml')):
        rule=yaml.safe_load(path.read_text()); validate_rule(rule); rules.append(rule)
    return rules

def match_condition(text):
    normalized=text.strip().casefold()
    aliases={'diabetes':'type-2-diabetes','ckd':'ckd-stage-3','pku':'phenylketonuria','msud':'maple-syrup-urine-disease','celiac':'celiac-disease','coeliac':'celiac-disease','nafld':'masld-fatty-liver','eating disorder':'eating-disorder-history','dialysis':'hemodialysis'}
    all_rules=catalog()
    for rule in all_rules:
        if normalized in (rule['name'].casefold(),rule['id']) or aliases.get(normalized)==rule['id']:
            return rule, normalized not in (rule['name'].casefold(),rule['id'])
    return None, False

def suggestions(text):
    rules=catalog(); names={r['name']:r for r in rules}
    return [names[n] for n in difflib.get_close_matches(text,list(names),n=3,cutoff=0.45)]

def bmi(profile):
    return round(profile.weight_kg/(profile.height_cm/100)**2,1)

def energy(profile):
    if profile.age<18 or profile.sex not in ('male','female'): return None
    bmr=10*profile.weight_kg+6.25*profile.height_cm-5*profile.age+(5 if profile.sex=='male' else -161)
    return round(bmr*{'sedentary':1.2,'light':1.375,'moderate':1.55,'active':1.725}[profile.activity])

def guidance(profile):
    selected=[]; unknown=[]; assumptions=[]
    for text in profile.conditions:
        rule,alias=match_condition(text)
        if rule: selected.append(rule)
        else: unknown.append(text)
        if alias: assumptions.append(f'Interpreted {text} as {rule["name"]}; confirm in selector')
    tier=max([r['tier'] for r in selected]+[2 if profile.age<18 or profile.pregnancy_trimester else 1])
    # Free text that implies a high-risk context must not enable weight-loss planning.
    risk_text=' '.join(unknown).lower()
    if any(w in risk_text for w in ['transplant','eating disorder','anorexi','bulimi','tube feed','pku','msud','icu']): tier=3
    if any(w in risk_text for w in ['pregnan','dialysis','cancer','chemotherap','child','ckd']): tier=max(2,tier)
    limits={}; conflicts=[]
    for rule in selected:
        for nutrient,target in rule['limits'].items():
            current=limits.setdefault(nutrient,{'minimum':None,'maximum':None,'unit':target['unit'],'sources':[]})
            for field in ('minimum','maximum'):
                if target.get(field) is not None:
                    old=current[field]; new=target[field]
                    current[field]=new if old is None else (max(old,new) if field=='minimum' else min(old,new))
                    if old is not None and old!=new: conflicts.append(f'{nutrient}: stricter {field} selected from condition rules')
            current['sources'].append(rule['name'])
    # No automatic energy restriction. Tier 2 requires individualized targets.
    if tier==1 and not unknown:
        kcal=energy(profile)
        if kcal: limits['kcal']={'minimum':None,'maximum':kcal,'unit':'kcal','sources':['Mifflin-St Jeor maintenance estimate; not a prescribed cap']}
    for nutrient,target in profile.targets.items():
        if nutrient not in UNITS or target.unit != UNITS[nutrient]: raise ValueError(f'Invalid target nutrient/unit: {nutrient}')
        if nutrient in limits: conflicts.append(f'{nutrient}: clinician target overrides condition default')
        limits[nutrient]={**target.model_dump(),'sources':['clinician-entered']}
    for nutrient,t in limits.items():
        if t['minimum'] is not None and t['maximum'] is not None and t['minimum']>t['maximum']:
            conflicts.append(f'Incompatible {nutrient} minimum and maximum; plan blocked pending clinician review')
    meal_targets={}
    for rule in selected:
        for nutrient,target in rule['per_meal_limits'].items():
            current=meal_targets.setdefault(nutrient,{'minimum':None,'maximum':None,'unit':target['unit']})
            if target.get('maximum') is not None:current['maximum']=min(current['maximum'],target['maximum']) if current['maximum'] is not None else target['maximum']
            if target.get('minimum') is not None:current['minimum']=max(current['minimum'],target['minimum']) if current['minimum'] is not None else target['minimum']
    for nutrient,target in profile.meal_targets.items():
        if nutrient not in UNITS or target.unit!=UNITS[nutrient]:raise ValueError(f'Invalid meal target nutrient/unit: {nutrient}')
        meal_targets[nutrient]=target.model_dump()
    banners=[]
    if tier>=2: banners.append('Consult your clinician before following this plan. Follow your dietitian’s plan.')
    if tier==3:
        banners.append('Clinician-only mode: food logging and avoid/allergen warnings only. No calorie restriction or weight-loss features.')
        # Targets remain in stored profile; never expose automated diet constraints for tier 3.
        limits={}
        meal_targets={}
    if unknown or any(not r['personalized'] for r in selected): banners.append('Not personalized for one or more conditions. Enter clinician targets.')
    if selected: banners.append('DRAFT: these rules have not been reviewed by a qualified clinician.')
    if len(selected)>1: conflicts.append('Multiple conditions: clinician review required; numeric limits alone cannot resolve every trade-off.')
    if profile.goal!='maintain': assumptions.append('Weight-change prescriptions are not generated by draft rules')
    return {'tier':tier,'bmi':bmi(profile) if profile.age>=18 and tier<3 else None,'targets':limits,'meal_targets':meal_targets,'conditions':selected,'unmatched':unknown,'banners':banners,'conflicts':conflicts,'assumptions':assumptions,'disclaimer':DISCLAIMER,'prefer':[x for r in selected for x in r['prefer']],'avoid':sorted(set(x for r in selected for x in r['avoid'])),'can_plan':tier==1 and not unknown}

def blocked_food(food,profile,g=None):
    g=g or guidance(profile)
    tokens=set(food['allergens']+food['tags']+[food['name'].lower(),food['id']])
    exclusions={x.strip().lower() for x in profile.allergies+profile.restrictions+g['avoid']}
    if 'gluten' in exclusions: exclusions.update(['wheat','barley','rye'])
    if 'milk' in exclusions: exclusions.add('dairy')
    if profile.diet=='vegan' and 'vegan' not in tokens: return 'Not vegan'
    if profile.diet=='vegetarian' and not {'vegan','vegetarian'}&tokens: return 'Not vegetarian'
    match=exclusions&tokens
    if match: return 'Avoid: '+', '.join(sorted(match))
    # Unrecognized allergy labels cannot be certified safe using this small catalog.
    known={'peanut','tree nut','milk','egg','wheat','soy','fish','shellfish','sesame','lactose','gluten','fructose','histamine','fodmap'}
    if any(a.lower() not in known for a in profile.allergies): return 'Custom allergy requires ingredient review'
    if {'fructose','histamine','fodmap'}&exclusions: return 'Intolerance requires individualized ingredient review'
    return None

def alerts(total,profile,food_ids=(),daily=None):
    g=guidance(profile); result=[]; catalog_foods=foods()
    for fid in food_ids:
        reason=blocked_food(catalog_foods[fid],profile,g)
        if reason: result.append({'level':'danger','message':f'{catalog_foods[fid]["name"]}: {reason}'})
    if g['tier']!=3:
        for scope,values in [('meal',total),('day',daily or {})]:
            for nutrient,target in g['targets'].items():
                value=values.get(nutrient)
                if value is None:
                    if nutrient in values: result.append({'level':'info','message':f'{nutrient}: missing data; cannot assess {scope} target'})
                    continue
                maximum=target.get('maximum')
                if maximum is not None and value['high']>maximum:
                    result.append({'level':'warn','message':f'{scope.title()} {nutrient} upper estimate {value["high"]:.0f} exceeds daily target {maximum:g}; review portion and uncertainty'})
        for nutrient,target in g['meal_targets'].items():
            if total.get(nutrient) and target.get('maximum') is not None and total[nutrient]['high']>target['maximum']:
                result.append({'level':'warn','message':f'Meal {nutrient} upper estimate exceeds the per-meal limit {target["maximum"]:g} {target["unit"]}'})
    drugs=' '.join(profile.medications).lower()
    interactions={'warfarin':'Keep vitamin K intake consistent; do not eliminate vegetables.','maoi':'Discuss tyramine-containing foods.','phenelzine':'Discuss tyramine-containing foods.','tranylcypromine':'Discuss tyramine-containing foods.','simvastatin':'Ask whether grapefruit must be avoided.','atorvastatin':'Ask whether grapefruit must be limited.','statin':'Grapefruit interactions depend on the specific statin.','metformin':'Discuss alcohol use.','levothyroxine':'Discuss timing with calcium, iron, soy and fiber.'}
    for drug,message in interactions.items():
        if drug in drugs: result.append({'level':'info','message':f'{message} Ask your doctor. Never change medication doses using this app.'})
    return result
