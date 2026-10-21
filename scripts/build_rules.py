"""Regenerate draft catalog. Never overwrites clinician-reviewed files."""
import json
from pathlib import Path
import re
import yaml
ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
'Metabolic / endocrine': 'Type 1 diabetes|Type 2 diabetes|Prediabetes|Gestational diabetes|Insulin resistance|Metabolic syndrome|Obesity|Underweight|Hypothyroidism|Hyperthyroidism|Hashimoto thyroiditis|Graves disease|PCOS|Cushing syndrome|Addison disease|Hyperparathyroidism|Hypoglycemia|Vitamin D deficiency|Vitamin B12 deficiency|Iron deficiency|Folate deficiency|Iodine deficiency|Calcium deficiency',
'Cardiovascular': 'Hypertension|Hypotension|Coronary artery disease|Heart failure|Atrial fibrillation|High cholesterol|High triglycerides|Atherosclerosis|Post myocardial infarction|Post stent|Post bypass|Stroke|Peripheral artery disease|Rheumatic heart disease',
'Kidney / urinary': 'CKD stage 1|CKD stage 2|CKD stage 3|CKD stage 4|CKD stage 5|Hemodialysis|Peritoneal dialysis|Kidney stones|Calcium oxalate stones|Calcium stones|Uric acid stones|Nephrotic syndrome|Recurrent UTI|Kidney transplant',
'Liver / GI / pancreas': 'MASLD fatty liver|Hepatitis|Cirrhosis|Gallstones|Pancreatitis|GERD|Gastritis|Peptic ulcer|IBS|Crohn disease|Ulcerative colitis|Celiac disease|Constipation|Diarrhea|Diverticulitis|Lactose intolerance|SIBO|Gastroparesis|Short bowel syndrome',
'Respiratory': 'Asthma|COPD|Cystic fibrosis|Sleep apnea|Post COVID',
'Bone / joint / muscle': 'Osteoporosis|Osteopenia|Osteoarthritis|Rheumatoid arthritis|Gout|Hyperuricemia|Sarcopenia|Fibromyalgia|Lupus',
'Blood': 'Iron deficiency anemia|B12 deficiency anemia|Folate deficiency anemia|Sickle cell disease|Thalassemia|Hemophilia|G6PD deficiency|Hemochromatosis|Anticoagulant use',
'Neurological / mental health': 'Epilepsy|Parkinson disease|Dementia|Migraine|Multiple sclerosis|Depression|Anxiety|ADHD|Autism food selectivity|Anorexia nervosa|Bulimia nervosa|Binge eating disorder|Eating disorder history',
'Cancer': 'Cancer treatment|Cancer before treatment|Cancer after treatment|Cancer cachexia|Mucositis|Cancer post surgery',
'Infectious / immune': 'HIV|Tuberculosis|Hepatitis B|Hepatitis C|Typhoid|Dengue recovery|Malaria recovery|Long COVID|Autoimmune condition',
'Skin / eye / ENT / dental': 'Eczema|Acne|Psoriasis|Diabetic retinopathy|Glaucoma|Macular degeneration|Dysphagia|Dental problems|Oral surgery',
'Reproductive / life stage': 'Pregnancy|Lactation|Menopause|Endometriosis|Infertility|Infancy|Childhood|Adolescence|Older adult',
'Genetic / rare metabolic': 'Phenylketonuria|Galactosemia|Maple syrup urine disease|Homocystinuria|Wilson disease|Familial hypercholesterolemia|Glycogen storage disease|Urea cycle disorder',
'Surgery / procedures': 'Bariatric sleeve surgery|Gastric bypass|Ostomy|GI surgery|Organ transplant|ICU recovery|Tube feeding|Bedridden recovery'
}
CODES = {'Type 1 diabetes':'E10','Type 2 diabetes':'E11','Prediabetes':'R73.03','Gestational diabetes':'O24.4','Obesity':'E66','Underweight':'R63.6','Hypertension':'I10','High cholesterol':'E78.0','High triglycerides':'E78.1','Coronary artery disease':'I25.1','Heart failure':'I50','Hypothyroidism':'E03.9','PCOS':'E28.2','Gout':'M10','Celiac disease':'K90.0','Lactose intolerance':'E73','Iron deficiency anemia':'D50','Osteoporosis':'M81','GERD':'K21','IBS':'K58','Kidney stones':'N20','Cirrhosis':'K74.6','Crohn disease':'K50','Ulcerative colitis':'K51','Epilepsy':'G40','Dysphagia':'R13','Phenylketonuria':'E70.0','Maple syrup urine disease':'E71.0','G6PD deficiency':'D55.0','Hemochromatosis':'E83.11'}
TIER1 = set('Type 1 diabetes|Type 2 diabetes|Prediabetes|Gestational diabetes|Hypertension|High cholesterol|Coronary artery disease|Obesity|Underweight|MASLD fatty liver|Gout|Celiac disease|Lactose intolerance|Iron deficiency anemia|Osteoporosis|Hypothyroidism|PCOS|GERD|IBS|Kidney stones'.split('|'))
TIER2 = set('Heart failure|Cirrhosis|Cancer treatment|Cancer cachexia|Pregnancy|Gestational diabetes|Infancy|Childhood|Adolescence|Crohn disease|Ulcerative colitis|Bariatric sleeve surgery|Gastric bypass|Epilepsy|Dysphagia|Hemodialysis|Peritoneal dialysis'.split('|'))
TIER3 = set('Phenylketonuria|Galactosemia|Maple syrup urine disease|Homocystinuria|Wilson disease|Glycogen storage disease|Urea cycle disorder|Kidney transplant|Organ transplant|ICU recovery|Tube feeding|Anorexia nervosa|Bulimia nervosa|Binge eating disorder|Eating disorder history'.split('|'))
SPECIAL = {
'Type 1 diabetes': (['Consistent carbohydrate amounts with clinician-directed meal timing; never adjust insulin using this app'],[],{},'ADA'),
'Type 2 diabetes': (['High-fiber minimally processed carbohydrates; distribute carbohydrates across meals'],['sugar-sweetened beverages'],{},'ADA'),
'Prediabetes': (['Whole grains, legumes and vegetables support glycemic management'],['sugar-sweetened beverages'],{},'ADA'),
'Gestational diabetes': (['Use the obstetric diabetes team’s carbohydrate and energy targets'],[],{},'ADA'),
'Hypertension': (['Vegetables and minimally processed meals support blood pressure management'],['high-sodium processed foods'],{'sodium_mg':{'maximum':2300,'unit':'mg'}},'AHA'),
'High cholesterol': (['Soluble fiber and unsaturated fats support lipid management'],['trans fats'],{},'AHA'),
'Coronary artery disease': (['Prefer unsaturated fats and whole foods'],['trans fats'],{'sodium_mg':{'maximum':2300,'unit':'mg'}},'AHA'),
'Obesity': (['Sustainable eating patterns; individualized energy goals'],[],{},'WHO'),
'Underweight': (['Energy-dense meals and adequate protein; evaluate cause of low weight'],[],{},'ESPEN'),
'MASLD fatty liver': (['Mediterranean-style eating and reduced sugar-sweetened drinks support management'],['alcohol'],{},'WHO'),
'Gout': (['Discuss hydration and individual triggers with clinician'],['organ meats','alcohol'],{},'ICMR-NIN'),
'Celiac disease': (['Certified gluten-free foods; check cross-contact'],['wheat','gluten','barley','rye'],{},'WHO'),
'Lactose intolerance': (['Lactose-free alternatives according to individual tolerance'],['lactose'],{},'ICMR-NIN'),
'Iron deficiency anemia': (['Iron-rich foods with vitamin C; investigate the cause with clinician'],[],{},'WHO'),
'Osteoporosis': (['Calcium-containing foods and clinician-directed vitamin D support bone health'],[],{},'WHO'),
'Hypothyroidism': (['Discuss levothyroxine timing with your prescriber'],[],{},'ICMR-NIN'),
'PCOS': (['Fiber-rich carbohydrates and regular meals support metabolic health'],[],{},'ICMR-NIN'),
'GERD': (['Smaller meals; avoid lying down soon after eating; identify personal triggers'],[],{},'ICMR-NIN'),
'IBS': (['Track individual triggers; low-FODMAP elimination needs supervision and reintroduction'],[],{},'ICMR-NIN'),
'Kidney stones': (['Determine stone type before changing calcium, oxalate or fluid intake'],[],{},'ICMR-NIN'),
'G6PD deficiency': (['Avoid fava beans'],['fava beans'],{},'WHO')
}
URLS={'ADA':'https://professional.diabetes.org/standards-of-care','AHA':'https://www.heart.org/en/healthy-living/healthy-eating','KDIGO':'https://kdigo.org/guidelines/ckd-evaluation-and-management/','ESPEN':'https://www.espen.org/guidelines-home/espen-guidelines','WHO':'https://www.who.int/news-room/fact-sheets/detail/healthy-diet','ICMR-NIN':'https://www.nin.res.in/dietaryguidelines/'}
def slug(name):
    return re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')
def main():
    for group, names in GROUPS.items():
        for name in names.split('|'):
            tier = 3 if name in TIER3 else 2 if name in TIER2 or name.startswith('CKD') else 1
            supported = name in SPECIAL
            prefer, avoid, limits, source = SPECIAL.get(name, (['Follow your clinician’s individualized nutrition plan'] if tier > 1 else ['Balanced varied meals support general health'],[],{},'KDIGO' if name.startswith('CKD') else 'WHO'))
            rule = {'id':slug(name),'name':name,'group':group,'version':'0.1.0','tier':tier,'status':'draft','personalized':supported,'icd10cm':CODES.get(name),'icd11':None,'coding_note':'Category-level ICD-10-CM where verified; unmapped entries need professional coding; never an inferred diagnosis','limits':limits,'per_meal_limits':{},'prefer':prefer,'avoid':avoid,'source':{'organization':source,'url':URLS[source],'verification':'Candidate source; rule not clinically reviewed'},'last_reviewed':None,'reviewer':None}
            path = ROOT/'app/rules'/f'{rule["id"]}.yaml'
            if path.exists() and yaml.safe_load(path.read_text()).get('status')=='reviewed':
                continue
            path.write_text(yaml.safe_dump(rule,sort_keys=False))
    template = {'id':'new-condition','name':'New condition','group':'Other','version':'0.1.0','tier':1,'status':'draft','personalized':False,'icd10cm':None,'icd11':None,'limits':{},'per_meal_limits':{},'prefer':['Balanced varied diet'],'avoid':[],'source':{'organization':'WHO','url':URLS['WHO'],'verification':'Not reviewed'},'last_reviewed':None,'reviewer':None}
    (ROOT/'data/rule-template.yaml').write_text(yaml.safe_dump(template, sort_keys=False))
if __name__=='__main__':
    main()
