import os
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from . import database as db
from .schemas import Profile, MealRequest
from .rules_engine import catalog, guidance, alerts, suggestions, DISCLAIMER
from .nutrition import foods, nutrients, interval, aggregate, split_nutrients, fdc_search
from .planner import plan
from .vision import load_image, analyze

app=FastAPI(title='NutriScope · local nutrition research',version='0.1.0')
STATIC=Path(__file__).parent/'static'
app.mount('/static',StaticFiles(directory=STATIC),name='static')

@app.middleware('http')
async def local_security(request,call_next):
    # Reject cross-origin writes and DNS rebinding. Bind to loopback by default.
    allowed=set(os.environ.get('NUTRI_ALLOWED_HOSTS','localhost,127.0.0.1,testserver').split(','))
    host=request.headers.get('host','').split(':')[0]
    if host not in allowed:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Host not allowed'},status_code=400)
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc!=request.headers.get('host'):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'Cross-origin writes are disabled'},status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    response.headers['Cache-Control']='no-store'
    response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    return response

@app.get('/')
def index(): return FileResponse(STATIC/'index.html')

@app.get('/api/health')
def health(): return {'status':'ok','mode':'research','disclaimer':DISCLAIMER,'privacy':'Profiles and history are stored locally in SQLite. Images are processed in memory and not retained.'}

@app.get('/api/conditions')
def conditions(q:str=''):
    return [r for r in catalog() if q.casefold() in (r['name']+' '+r['group']+' '+str(r['icd10cm'])).casefold()]

@app.get('/api/conditions/suggest')
def suggest(q:str): return suggestions(q)

@app.get('/api/foods')
def food_list(): return list(foods().values())

@app.post('/api/profiles')
def create_profile(profile:Profile):
    try: result=guidance(profile)
    except ValueError as e: raise HTTPException(422,str(e)) from e
    db.save_profile(profile)
    return {'profile':profile.model_dump(),'guidance':result}

def required_profile(profile_id):
    p=db.get_profile(profile_id)
    if p is None: raise HTTPException(404,'Create a local profile first')
    return p

@app.get('/api/profiles/{profile_id}')
def read_profile(profile_id:str):
    p=required_profile(profile_id)
    return {'profile':p.model_dump(),'guidance':guidance(p)}

@app.delete('/api/profiles/{profile_id}')
def remove_profile(profile_id:str):
    db.delete_profile(profile_id)
    return {'deleted':True}

@app.get('/api/profiles/{profile_id}/plan')
def meal_plan(profile_id:str): return plan(required_profile(profile_id))

@app.get('/api/profiles/{profile_id}/history')
def history(profile_id:str):
    required_profile(profile_id)
    return db.history(profile_id)

@app.post('/api/meals/analyze')
def meal_analysis(request:MealRequest):
    p=required_profile(request.profile_id); result=[]
    for item in request.items:
        if item.food_id not in foods(): raise HTTPException(422,f'Unknown food {item.food_id}; confirm a catalog match')
        uncertainty=max(item.uncertainty,.1 if item.basis=='measured' else .35)
        result.append({**item.model_dump(),'uncertainty':uncertainty,'name':foods()[item.food_id]['name'],'weight_g':interval(item.grams,uncertainty),'nutrients':nutrients(item.food_id,item.grams,uncertainty),'assumptions':['Food composition varies; interval is illustrative, not statistically calibrated',foods()[item.food_id]['source']]})
    total=aggregate([x['nutrients'] for x in result])
    today=datetime.now(timezone.utc).date().isoformat()
    prior=[x['nutrients'] for x in db.history(p.id) if x['created_at'].startswith(today)]
    daily=aggregate(prior+[total])
    body={'slot':request.slot,'items':result,'nutrients':total,'portions':split_nutrients(total,request.percentages),'daily_with_meal':daily,'alerts':alerts(total,p,[x.food_id for x in request.items],daily),'assumptions':['Daily totals use UTC dates; only logged meals are counted']}
    return body

@app.post('/api/meals')
def log_meal(request:MealRequest):
    result=meal_analysis(request)
    result['id']=db.save_meal(request.profile_id,result)
    return result

@app.post('/api/images/analyze')
async def image_analysis(file:UploadFile=File(...),reference_cm:float|None=Form(None),reference_pixels:float|None=Form(None)):
    content=await file.read(10_000_001)
    try:
        image=load_image(content)
        if reference_cm is not None and reference_cm<=0: raise ValueError('Reference dimension must be positive')
        if reference_pixels is not None and reference_pixels<=0: raise ValueError('Reference pixel span must be positive')
        return await run_in_threadpool(analyze,image,reference_cm,reference_pixels)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    except (ImportError,FileNotFoundError,KeyError,OSError) as exc:
        raise HTTPException(503,'Model artifact configuration is incomplete. Use manual logging; see models/README.md.') from exc

@app.get('/api/fdc/search')
async def search_fdc(q:str):
    if not q.strip() or len(q)>150: raise HTTPException(422,'Search must have 1–150 characters')
    key=os.environ.get('FDC_API_KEY')
    if not key: return {'source':'local-fallback','foods':[f for f in foods().values() if q.lower() in f['name'].lower()]}
    import httpx
    try: return {'source':'USDA FDC','data':await fdc_search(q,key)}
    except httpx.HTTPError:
        return {'source':'local-fallback','warning':'FDC unavailable','foods':[f for f in foods().values() if q.lower() in f['name'].lower()]}
