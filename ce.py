import json
import os

os.chdir('data')

def translate(text, type):
    mapping = json.loads(open(f"names/{type}.json", "r").read())
    if str(text) in mapping:
        if mapping[str(text)]:
            return mapping[str(text)]
    return str(text)

ces = []

for i in range(1, 7):
    ces += json.loads(open(f"chaldea-data/dist/craftEssences.{i}.json", "r").read())

def find_ce(id):
    for ce in ces:
        if ce['id'] == id:
            return ce
    return None

def get_ce(id, filters):
    raw = find_ce(id)
    # print(raw['name'])
    return {
        "id": raw['id'],
        "name": translate(raw['name'], "ce"),
        "img": dict(raw['extraAssets']['equipFace']['equip']).popitem()[1],
        "cost": raw['cost'],
        "filters": filters
    }

ce_data = []
ce_data.append(get_ce(9408800, [([304], 20),([203], 20)]))
ce_data.append(get_ce(9408590, [([300, 2], 20)]))
ce_data.append(get_ce(9408390, [([2780], 20)]))
ce_data.append(get_ce(9408220, [([103], 20)]))
ce_data.append(get_ce(9408060, [([104], 20)]))
ce_data.append(get_ce(9407850, [([300, 303], 20)]))
ce_data.append(get_ce(9407740, [([2654], 20)]))
ce_data.append(get_ce(9308100, [([2883], 10)]))
ce_data.append(get_ce(9407480, [([2821], 20)]))
ce_data.append(get_ce(9406740, [([], 2.5)]))
ce_data.append(get_ce(9406200, [([], 2.5)]))
ce_data.append(get_ce(9406010, [([], 2.5)]))
ce_data.append(get_ce(9405170, [([], 5)]))
ce_data.append(get_ce(9404360, [([], 5)]))
ce_data.append(get_ce(9404180, [([], 5)]))
ce_data.append(get_ce(9403990, [([], 5)]))
ce_data.append(get_ce(9403520, [([], 5)]))
ce_data.append(get_ce(9401970, [([], 10)]))
ce_data.append(get_ce(9400980, [([], -50)]))
ce_data.append(get_ce(9408990, [([301, 2858], 20)]))
ce_data.append(get_ce(9311320, [([], 2)]))
ce_data.append(get_ce(9311450, [([], 2)]))

open('ces.json','w').write(json.dumps(ce_data, ensure_ascii=False, indent=4))

# print(get_ce(9311450, [([], 2)]))