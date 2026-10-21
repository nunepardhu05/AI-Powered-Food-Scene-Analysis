"""Constrained illustrative menus; never a clinical prescription."""
from itertools import product
import numpy as np
from scipy.optimize import linprog
from .nutrition import foods,nutrients,aggregate,household
from .rules_engine import guidance,blocked_food
SLOTS=[('breakfast',.22),('mid-morning snack',.10),('lunch',.28),('evening snack',.10),('dinner',.30)]
# Portions describe edible cooked components, not inferred recipes or hidden oil/salt.
MENUS={
 'breakfast':[
  ('Chapati, lentils and apple',{'chapati':70,'lentils':100,'apple':80},'Indian'),
  ('Oatmeal, apple and yogurt',{'oats':200,'apple':100,'yogurt':100},'International'),
  ('Rice, tofu and broccoli',{'rice':100,'tofu':75,'broccoli':100},'International'),
  ('Egg, chapati and apple',{'egg':100,'chapati':60,'apple':100},'Indian')],
 'mid-morning snack':[
  ('Apple and yogurt',{'apple':100,'yogurt':100},'International'),
  ('Apple and peanuts',{'apple':100,'peanut':20},'International'),
  ('Apple and boiled egg',{'apple':100,'egg':50},'International'),
  ('Apple and plain tofu',{'apple':100,'tofu':50},'International')],
 'lunch':[
  ('Rice, lentils and broccoli',{'rice':160,'lentils':150,'broccoli':100},'Indian'),
  ('Chapati, lentils and broccoli',{'chapati':90,'lentils':150,'broccoli':100},'Indian'),
  ('Chicken, rice and broccoli',{'chicken':100,'rice':160,'broccoli':100},'International'),
  ('Tofu, rice and broccoli',{'tofu':100,'rice':160,'broccoli':100},'International')],
 'evening snack':[
  ('Yogurt and apple',{'yogurt':125,'apple':80},'International'),
  ('Peanuts and apple',{'peanut':20,'apple':80},'International'),
  ('Lentils and broccoli',{'lentils':100,'broccoli':50},'Indian'),
  ('Tofu and broccoli',{'tofu':80,'broccoli':80},'International')],
 'dinner':[
  ('Chapati with lentils and broccoli',{'chapati':90,'lentils':150,'broccoli':100},'Indian'),
  ('Rice with lentils and broccoli',{'rice':180,'lentils':150,'broccoli':100},'Indian'),
  ('Chicken with rice and broccoli',{'chicken':120,'rice':160,'broccoli':100},'International'),
  ('Tofu with rice and broccoli',{'tofu':120,'rice':160,'broccoli':100},'International')]
}

def meal(menu,scale,catalog):
    title,components,region=menu;entries=[]
    for fid,base in components.items():
        grams=base*scale;food=catalog[fid]
        entries.append({'food_id':fid,'name':food['name'],'grams':round(grams,2),'weight_range_g':{'low':round(grams*.9,2),'high':round(grams*1.1,2)},'household':household(grams,food['density'],food['packing']),'nutrients':nutrients(fid,grams,.1)})
    return {'name':title,'cuisine':region,'items':entries,'nutrients':aggregate([i['nutrients'] for i in entries])}

def respects(totals,targets):
    for nutrient,t in targets.items():
        value=totals.get(nutrient)
        if value is None:return False
        if t.get('maximum') is not None and value['high']>t['maximum']+.02:return False
        if t.get('minimum') is not None and value['low']<t['minimum']-.02:return False
    return True

def plan(profile):
    g=guidance(profile)
    if not g['can_plan']:
        return {'status':'clinician_required','meals':[],'reason':'Automated planning unavailable for supervised, clinician-only or unmatched conditions. Log foods and use your clinician’s plan.','guidance':g}
    if any('Incompatible' in c for c in g['conflicts']):
        return {'status':'infeasible','meals':[],'reason':'Conflicting nutrient bounds require clinical review.'}
    catalog=foods();allowed={fid:f for fid,f in catalog.items() if not blocked_food(f,profile,g)}
    energy=g['targets'].get('kcal',{}).get('maximum')
    if energy is None:return {'status':'needs_targets','meals':[],'reason':'Enter an individualized energy reference to create an illustrative menu.'}
    choices=[]
    for slot,_ in SLOTS:
        menus=[m for m in MENUS[slot] if all(fid in allowed for fid in m[1])]
        menus.sort(key=lambda m:m[2].casefold()!=profile.region.casefold())
        if not menus:return {'status':'infeasible','meals':[],'reason':f'Insufficient verified {slot} options for these restrictions.'}
        choices.append(menus)
    targets=g['targets'];selected=None;selected_scales=None
    for combination in product(*choices):
        bases=[meal(m,1,catalog)['nutrients'] for m in combination]
        if any(any(b.get(k) is None for b in bases) for k in targets):continue
        matrix=[];bounds=[]
        for nutrient,t in targets.items():
            if t.get('maximum') is not None:matrix.append([b[nutrient]['high'] for b in bases]);bounds.append(t['maximum'])
            if t.get('minimum') is not None:matrix.append([-b[nutrient]['low'] for b in bases]);bounds.append(-t['minimum'])
        for index,(_,fraction) in enumerate(SLOTS):
            row=np.zeros(5);row[index]=-bases[index]['kcal']['value'];matrix.append(row);bounds.append(-energy*.85*fraction)
            row=np.zeros(5);row[index]=bases[index]['kcal']['value'];matrix.append(row);bounds.append(energy*.91*fraction)
        missing_meal_data=False
        for index in range(5):
            for nutrient,t in g['meal_targets'].items():
                if bases[index].get(nutrient) is None:missing_meal_data=True;continue
                if t.get('maximum') is not None:
                    row=np.zeros(5);row[index]=bases[index][nutrient]['high'];matrix.append(row);bounds.append(t['maximum'])
                if t.get('minimum') is not None:
                    row=np.zeros(5);row[index]=-bases[index][nutrient]['low'];matrix.append(row);bounds.append(-t['minimum'])
        if missing_meal_data:continue
        result=linprog(np.ones(5),A_ub=np.array(matrix),b_ub=np.array(bounds),bounds=[(.25,3)]*5,method='highs')
        if result.success:
            selected=combination;selected_scales=result.x;break
    if selected is None:
        return {'status':'infeasible','meals':[],'reason':'No menu satisfies all available bounds with this small catalog. Missing nutrient data, severe limits or limited recipe variety may prevent a plan.'}
    meals=[dict(slot=slot,**meal(m,float(scale),catalog)) for (slot,_),m,scale in zip(SLOTS,selected,selected_scales)]
    for i,entry in enumerate(meals):
        swaps=[]
        for candidate in choices[i]:
            if candidate==selected[i]:continue
            base=meal(candidate,1,catalog)
            scale=entry['nutrients']['kcal']['value']/base['nutrients']['kcal']['value']
            if not .25<=scale<=3:continue
            alternative=meal(candidate,scale,catalog)
            changed=[m['nutrients'] if j!=i else alternative['nutrients'] for j,m in enumerate(meals)]
            if respects(aggregate(changed),targets) and respects(alternative['nutrients'],g['meal_targets']):swaps.append(alternative)
        entry['swaps']=swaps[:3]
        entry['swap_notice']='Each swap was checked individually against this full day. Multiple simultaneous swaps need recalculation.' if swaps else 'No swap satisfies all current constraints; do not substitute without recalculation.'
    totals=aggregate([m['nutrients'] for m in meals])
    if not respects(totals,targets):return {'status':'infeasible','meals':[],'reason':'Numerical validation failed; no plan issued.'}
    return {'status':'illustrative','meals':meals,'totals':totals,'assumptions':['Small illustrative Indian/international menu library; other regional cuisines use these fallback options.','Cooked components exclude added oil, salt, sauces and hidden ingredients; log them separately.','10% nutrient uncertainty is illustrative, not calibrated.','No safety guarantee for cross-contact; verify package labels.','Maintenance-energy reference is approximate; this is not a weight-loss prescription.'],'guidance':g}
