import io
from PIL import Image
import pytest
from app import database as db

def profile(client,**extra):
    response=client.post('/api/profiles',json={'sex':'female',**extra})
    assert response.status_code==200
    return response.json()


def test_api_health_and_page(client):
    assert client.get('/').status_code==200
    assert client.get('/api/health').json()['status']=='ok'
    assert client.get('/api/conditions?q=diabetes').json()
    assert client.get('/api/foods').json()


def test_empty_profile(client):
    assert client.get('/api/profiles/local').status_code==404
    assert client.post('/api/meals',json={'items':[{'food_id':'rice','grams':100}]}).status_code==404


def test_database_roundtrip_delete(client):
    profile(client,name='Test person',clinical={'HbA1c_percent':7.2})
    p=client.get('/api/profiles/local').json()['profile']
    assert p['name']=='Test person' and p['clinical']['HbA1c_percent']==7.2
    assert client.post('/api/meals',json={'items':[{'food_id':'rice','grams':100}]}).status_code==200
    assert len(client.get('/api/profiles/local/history').json())==1
    assert client.delete('/api/profiles/local').status_code==200
    assert db.history('local')==[]


def test_meal_preview_does_not_save(client):
    profile(client)
    result=client.post('/api/meals/analyze',json={'items':[{'food_id':'rice','grams':200}],'percentages':[50,50]}).json()
    assert result['nutrients']['kcal']['value']==260
    assert len(result['portions'])==2
    assert client.get('/api/profiles/local/history').json()==[]


def test_daily_accumulation(client):
    profile(client,targets={'sodium_mg':{'maximum':500,'unit':'mg'}})
    request={'items':[{'food_id':'chapati','grams':100}]}
    client.post('/api/meals',json=request)
    result=client.post('/api/meals/analyze',json=request).json()
    assert result['daily_with_meal']['sodium_mg']['value']==800
    assert any('Day sodium' in a['message'] for a in result['alerts'])


def test_unknown_food_and_bad_inputs(client):
    profile(client)
    assert client.post('/api/meals',json={'items':[{'food_id':'unknown','grams':100}]}).status_code==422
    assert client.post('/api/meals',json={'items':[]}).status_code==422
    assert client.post('/api/meals',json={'items':[{'food_id':'rice','grams':100}],'percentages':[0,100]}).status_code==422
    assert client.post('/api/profiles',json={'targets':{'sodium_mg':{'maximum':2,'unit':'g'}}}).status_code==422


def test_security(client):
    assert client.post('/api/profiles',json={},headers={'Origin':'https://evil.example'}).status_code==403
    assert client.get('/api/health',headers={'Host':'evil.example'}).status_code==400
    assert 'frame-ancestors' in client.get('/').headers['content-security-policy']


def test_image_route_and_manual_completion(client):
    profile(client,allergies=['peanut'])
    buf=io.BytesIO();Image.new('RGB',(100,100),'white').save(buf,format='PNG')
    vision=client.post('/api/images/analyze',files={'file':('plate.png',buf.getvalue(),'image/png')})
    assert vision.status_code==200
    assert vision.json()['status']=='models_unavailable'
    result=client.post('/api/meals',json={'items':[{'food_id':'peanut','grams':30,'basis':'measured'}]}).json()
    assert result['items'][0]['weight_g']['low']==27
    assert result['nutrients']['kcal']['value']==176.1
    assert any(a['level']=='danger' for a in result['alerts'])


def test_mocked_detection_geometry_nutrition_integration(client,monkeypatch):
    """Contract integration with synthetic detections; NOT model accuracy evidence."""
    import numpy as np
    from app import main
    from app.volume import estimate_volume,physics_weight
    def synthetic(image,reference_cm,reference_pixels):
        mask=np.zeros((20,20),bool);mask[5:15,5:15]=True
        depth=np.full((20,20),100.);depth[mask]=98.
        volume=estimate_volume(depth,mask,np.ones_like(mask),1,calibrated=True)['volume_ml']
        weight=physics_weight(volume,.75,.9)
        return {'status':'synthetic_test_only','items':[{'food_id':'rice','weight_g':weight}],'warnings':['Synthetic fixture; not an ML prediction']}
    monkeypatch.setattr(main,'analyze',synthetic)
    profile(client)
    buf=io.BytesIO();Image.new('RGB',(20,20),'white').save(buf,format='PNG')
    detected=client.post('/api/images/analyze',files={'file':('fixture.png',buf.getvalue(),'image/png')}).json()['items'][0]
    result=client.post('/api/meals/analyze',json={'items':[{'food_id':detected['food_id'],'grams':detected['weight_g']['value']}]}).json()
    assert detected['weight_g']['value']==pytest.approx(135)
    assert result['nutrients']['kcal']['value']==pytest.approx(175.5)
    assert 'alerts' in result


def test_fdc_fallback(client,monkeypatch):
    monkeypatch.delenv('FDC_API_KEY',raising=False)
    result=client.get('/api/fdc/search?q=rice').json()
    assert result['source']=='local-fallback' and result['foods']
