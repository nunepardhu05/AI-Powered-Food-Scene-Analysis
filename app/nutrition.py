import csv
import json
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
NUTRIENTS = ['kcal','carbs_g','sugars_g','protein_g','fat_g','fiber_g','sodium_mg','potassium_mg','phosphorus_mg','iron_mg','calcium_mg','vitamin_c_mg','vitamin_k_ug','vitamin_b12_ug','vitamin_d_ug','purines_mg','oxalate_mg']

def foods():
    with (ROOT/'data/foods.csv').open() as f:
        result = {}
        for row in csv.DictReader(f):
            for key in NUTRIENTS + ['density','packing','gi']:
                row[key] = float(row[key]) if row[key] else None
            row['allergens'] = [x for x in row['allergens'].split(';') if x]
            row['tags'] = row['tags'].split(';')
            result[row['id']] = row
        return result

def nutrients(food_id, grams, uncertainty=0.1):
    if grams <= 0 or grams > 5000 or not 0 <= uncertainty <= 1:
        raise ValueError('Invalid portion or uncertainty')
    food = foods()[food_id]
    values = {}
    for key in NUTRIENTS:
        val = food[key]
        values[key] = None if val is None else interval(val * grams / 100, uncertainty)
    values['glycemic_load'] = None if food['gi'] is None else interval(food['gi'] / 100 * max(0,food['carbs_g']-food['fiber_g']) * grams/100, uncertainty)
    return values

def interval(value, relative):
    return {'value': round(value,3), 'low': round(max(0,value*(1-relative)),3), 'high': round(value*(1+relative),3)}

def aggregate(items):
    keys = list(NUTRIENTS) + ['glycemic_load']
    return {k: None if any(item.get(k) is None for item in items) else {bound: round(sum(item[k][bound] for item in items),3) for bound in ['value','low','high']} for k in keys}

def split_nutrients(total, percentages):
    if not percentages or any(x<=0 for x in percentages) or abs(sum(percentages)-100)>0.001:
        raise ValueError('Portions must be positive and sum to 100')
    return [{k: None if v is None else {b:round(n*p/100,3) for b,n in v.items()} for k,v in total.items()} for p in percentages]

def household(grams, density=1, packing=1):
    if min(grams,density,packing)<=0:
        raise ValueError('Positive quantities required')
    ml = grams/(density*packing)
    return {'ml':round(ml,1),'cups_240ml':round(ml/240,2),'katori_150ml':round(ml/150,2),'assumption':'240 ml cup; 150 ml katori; household sizes and packing vary'}

def decompose(dish, grams):
    recipes = json.loads((ROOT/'data/recipes.json').read_text())
    recipe = recipes[dish]
    return {'items':[{'food_id':k,'grams':grams*v} for k,v in recipe['ingredients'].items()], 'estimated':True, 'assumptions':[recipe['note']]}

async def fdc_search(query, api_key):
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get('https://api.nal.usda.gov/fdc/v1/foods/search',params={'query':query,'api_key':api_key,'pageSize':10})
        response.raise_for_status()
        return response.json()
