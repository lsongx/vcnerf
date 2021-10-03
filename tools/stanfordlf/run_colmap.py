import os
import shutil
from tools.nex_pose.imgs2poses import colmapGenPoses

scenes = ['amethyst', 'cards-big', 'chess', 'jellybeans', 'lego-gantry', 'lego-truck', 'treasure', 
          'bracelet', 'cards-small', 'eucalyptus-flowers', 'lego-bulldozer', 'lego-knights', 'stanfordbunny']
scenes = ['lego-knights']
base_dir = os.path.expanduser('~/data/3d/StanfordLF')
create_dir = base_dir.replace('StanfordLF', 'StanfordLF-colmap')
if not os.path.isdir(create_dir):
    os.mkdir(create_dir)

for scene in scenes:
    ori_path = os.path.join(base_dir, scene)
    new_path = os.path.join(create_dir, scene)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
        shutil.copytree(ori_path, os.path.join(new_path, 'images'))

    colmapGenPoses(new_path)
    print(f'{scene} finished')

