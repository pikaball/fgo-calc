import os
import json
import time
import requests
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__))+'/data')

def fetch_git_repo():
    last_update = 0
    if os.path.exists("update.txt"):
        last_update = int(open("update.txt").read().strip())
    if time.time() - last_update < 86400:
        print("Last update was less than 24 hours ago. Skipping fetch.")
        return

    try:
        if os.path.exists("chaldea-data"):
            os.chdir("chaldea-data")
            os.system("git pull")
            os.chdir("..")
        else:
            os.system("git clone https://github.com/chaldea-center/chaldea-data")
        open("update.txt", "w").write(str(int(time.time())))
        print("Git repo fetched successfully.")
    except Exception as e:
        print(f"Error fetching git repo: {e}")
        print("Retrying in 10 seconds...")
        time.sleep(10)
        fetch_git_repo()

fetch_git_repo()

def get_translation():
    if not os.path.exists("names"):
        os.makedirs("names")

    traits_raw = json.loads(open("chaldea-data/mappings/trait.json", "r").read())
    traits = {}
    for k, v in traits_raw.items():
        traits[k] = v["CN"]
    open("names/traits.json", "w").write(json.dumps(traits, ensure_ascii=False, indent=4))

    ce_raw = json.loads(open("chaldea-data/mappings/ce_names.json", "r").read())
    ce = {}
    for k, v in ce_raw.items():
        ce[k] = v["CN"]
    open("names/ce.json", "w").write(json.dumps(ce, ensure_ascii=False, indent=4))

    servant_raw = json.loads(open("chaldea-data/mappings/svt_names.json", "r").read())
    servant = {}
    for k, v in servant_raw.items():
        servant[k] = v["CN"]
    open("names/servant.json", "w").write(json.dumps(servant, ensure_ascii=False, indent=4))

    costume_raw = json.loads(open("chaldea-data/mappings/costume_names.json", "r").read())
    costume = {}
    for k, v in costume_raw.items():
        costume[k] = v["CN"]
    open("names/costume.json", "w").write(json.dumps(costume, ensure_ascii=False, indent=4))

get_translation()

def find_files(name):
    files = []
    for root, dirs, filenames in os.walk("chaldea-data/dist"):
        for filename in filenames:
            if filename.startswith(name + ".") and filename.endswith(".json"):
                files.append(os.path.join(root, filename))
    return files

def translate(text, type):
    mapping = json.loads(open(f"names/{type}.json", "r").read())
    if str(text) in mapping:
        return mapping[str(text)]
    return text

def get_traits(trait_list):
    traits = []
    for trait in trait_list:
        traits.append(trait['id'])
    return traits

def process_servant(test):
    data = {}
    data['id'] = test['id']

    data['name'] = translate(test['name'], "servant")
    # data['img'] = test['extraAssets']['faces']['costume']

    traits = get_traits(test['traits'])
    cost = test['cost']
    img = test['extraAssets']['faces']['ascension']['1']

    data['diff'] = {}

    data['diff']['default'] = {
        'name': '默认',
        'traits': traits,
        'img': img,
        'cost': cost
    }

    data['diff']['asc1'] = {
        'name': "灵基再临1",
        'traits': traits,
        'img': test['extraAssets']['faces']['ascension']['2'],
        'cost': cost
    }

    data['diff']['asc2'] = {
        'name': "灵基再临2",
        'traits': traits,
        'img': test['extraAssets']['faces']['ascension']['3'],
        'cost': cost
    }

    data['diff']['asc3'] = {
        'name': "灵基再临3",
        'traits': traits,
        'img': test['extraAssets']['faces']['ascension']['4'],
        'cost': cost
    }

    if 'costume' in test['extraAssets']['faces']:
        for key, value in test['extraAssets']['faces']['costume'].items():
            data['diff'][key] = {
                'name': translate(test['profile']['costume'][key]['name'], "costume"),
                'traits': traits,
                'img': value,
                'cost': cost
            }

    costume_map = {}
    if 'costume' in test['profile']:
        for key, value in test['profile']['costume'].items():
            costume_map[str(value['id'])] = str(key)

    if 'overwriteCost' in test['ascensionAdd']:
        oc = test['ascensionAdd']['overwriteCost']
        if 'costume' in oc:
            for key, value in oc['costume'].items():
                data['diff'][costume_map[str(key)]]['cost'] = value
        if 'ascension' in oc:
            for key, value in oc['ascension'].items():
                asc_key = f"asc{key}"
                if asc_key in data['diff']:
                    data['diff'][asc_key]['cost'] = value

    if 'individuality' in test['ascensionAdd']:
        indiv = test['ascensionAdd']['individuality']
        if 'ascension' in indiv:
            for key, value in indiv['ascension'].items():
                asc_key = f"asc{key}"
                if key == '0':
                    asc_key = 'default'
                if asc_key in data['diff']:
                    data['diff'][asc_key]['traits'] = get_traits(value)
        if 'costume' in indiv:
            for key, value in indiv['costume'].items():
                if str(key) in data['diff']:
                    data['diff'][str(key)]['traits'] = get_traits(value)

    # 去重
    diffs = []

    for k in list(data['diff'].keys()):
        diffflag = True
        for diffkey in diffs:
            if data['diff'][k]['traits'] == data['diff'][diffkey]['traits'] and data['diff'][k]['cost'] == data['diff'][diffkey]['cost']:
                del data['diff'][k]
                diffflag = False
                break
        if diffflag:
            diffs.append(k)

    return data

processed = []

remove_list = [2501500]

for file in find_files("servants"):
    raw = json.loads(open(file, "r").read())
    for servant in raw:
        try:
            if servant['id'] in remove_list:
                continue
            processed.append(process_servant(servant))
        except Exception as e:
            print(f"Error processing servant {servant['name']}: {e}")

open('servants.json','w').write(json.dumps(processed, ensure_ascii=False, indent=4))

os.chdir('..')

print('[+] Servant info update done.')