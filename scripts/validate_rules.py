from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.rules_engine import catalog
if __name__=='__main__':
    rules=catalog()
    ids=[r['id'] for r in rules]
    assert len(ids)==len(set(ids)), 'Duplicate rule IDs'
    print(f'Validated {len(rules)} rule files. Draft rules are NOT clinically validated.')
