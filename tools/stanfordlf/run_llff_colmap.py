import os
import shutil
import logging
import sys
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../llff_pose/')
from pose_utils import gen_poses

scenes = ['amethyst', 'cards-big', 'chess', 'jellybeans', 'lego-gantry', 'lego-truck', 'treasure', 
          'bracelet', 'cards-small', 'eucalyptus-flowers', 'lego-bulldozer', 'lego-knights', 'stanfordbunny']
# scenes = ['lego-knights']
base_dir = os.path.expanduser('~/data/3d/StanfordLF')
create_dir = base_dir.replace('StanfordLF', 'StanfordLF-colmap')
if not os.path.isdir(create_dir):
    os.mkdir(create_dir)

handlers = [logging.StreamHandler()]
logfile = f'colmap-log.txt'
handlers.append(logging.FileHandler(os.path.join(create_dir, logfile), mode='w'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-5s %(message)s',
    datefmt='%m-%d %H:%M:%S', handlers=handlers,
)

views = ['_00_00_', '_00_16_', '_08_08_', '_16_00_', '_16_16_']
# views = ['_00_00_', '_00_16_', '_08_08_', '_16_00_', '_16_16_',
#          '_04_08_', '_10_08_',]
#         #  '_04_08_', '_10_08_', '_08_04_', '_08_10_',]
def is_in_views(view, views=views):
    for v in views:
        if v in view:
            return True
    return False

good, error = [], []
for scene in scenes:
    ori_path = os.path.join(base_dir, scene, 'rectified')
    # ori_path = os.path.join(base_dir, scene, 'sparse')
    new_path = os.path.join(create_dir, scene)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
        logging.info(f'{scene} copy images')
        # shutil.copytree(ori_path, os.path.join(new_path, 'images'))

        os.mkdir(os.path.join(new_path, 'images'))
        for f in os.listdir(ori_path):
            if is_in_views(f):
                shutil.copy(os.path.join(ori_path, f), os.path.join(new_path, 'images', f))
                logging.info(f'{f} copied')

        logging.info(f'{scene} copy finished')

    try:
        logging.info(f'>>> {scene} started')
        # colmapGenPoses(new_path)
        # gen_poses(new_path, 'exhaustive_matcher')
        gen_poses(new_path, 'sequential_matcher')
        logging.info(f'>>> {scene} finished')
        good.append(scene)
    except:
        logging.info(f'>>> {scene} error')
        error.append(scene)

logging.info(f'good scene: {good}')
logging.info(f'error scene: {error}')
