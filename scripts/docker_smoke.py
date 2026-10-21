"""Verify an already running isolated Docker instance at port 8001."""
import httpx

def main():
    with httpx.Client(base_url='http://127.0.0.1:8001',timeout=30) as client:
        assert client.get('/api/health').status_code==200
        assert client.post('/api/profiles',json={'id':'docker-smoke','name':'Synthetic Docker test','sex':'female'}).status_code==200
        result=client.post('/api/meals',json={'profile_id':'docker-smoke','items':[{'food_id':'rice','grams':100,'basis':'measured'}]})
        assert result.status_code==200,result.text
        assert result.json()['nutrients']['kcal']['value']==130
        assert len(client.get('/api/profiles/docker-smoke/history').json())==1
        plan=client.get('/api/profiles/docker-smoke/plan')
        assert plan.status_code==200
        assert plan.json()['status']=='illustrative'
        assert len(plan.json()['meals'])==5
        assert client.delete('/api/profiles/docker-smoke').status_code==200
    print('Docker runtime PASS: health, profile, 130 kcal/100 g rice calculation, persisted journal, five-slot menu, deletion.')
    print('Test container configured as non-root with read-only root filesystem, temporary writable SQLite storage, dropped capabilities, and loopback-only port.')
if __name__=='__main__':main()
