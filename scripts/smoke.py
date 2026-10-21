"""Three generated image fixtures. Does not assert real food recognition accuracy."""
from pathlib import Path
import io
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from app.vision import load_image,analyze

def main():
    out=Path(__file__).resolve().parents[1]/'data/samples';out.mkdir(exist_ok=True)
    for name,color in [('empty_plate',(235,235,225)),('dark',(5,5,5)),('synthetic_food_scene',(210,220,205))]:
        image=Image.new('RGB',(320,240),color)
        if name!='dark':
            draw=ImageDraw.Draw(image);draw.ellipse((60,20,260,220),fill='white',outline='gray',width=3)
            if name=='synthetic_food_scene':
                draw.ellipse((85,60,170,150),fill=(160,180,100));draw.ellipse((160,100,230,180),fill=(210,160,100))
        path=out/f'{name}.png';image.save(path)
        result=analyze(load_image(path.read_bytes()))
        assert result['status']=='models_unavailable' and not result['items']
        print(f'{name}.png: PASS — no fabricated detections; {len(result["warnings"])} warnings')
    print('3/3 fixture smoke checks passed. Real-model smoke evaluation has NOT run.')
if __name__=='__main__':main()
