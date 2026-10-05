"""Check Python and notebook code syntax without executing experiments."""
from pathlib import Path
import ast,json
for root in ('src','scripts','tests'):
    for p in Path(root).rglob('*.py'): ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
for p in Path('notebooks').glob('*.ipynb'):
    notebook=json.loads(p.read_text(encoding='utf-8'))
    for i,cell in enumerate(notebook['cells']):
        assert cell.get('id'), f'{p}: missing cell ID'
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']),filename=f'{p}:cell {i}')
print('Python and notebook syntax validation passed')
