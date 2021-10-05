import os
import shutil
import sys
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/../nex_pose/')
from imgs2poses import colmapGenPoses

scenes = ['amethyst', 'cards-big', 'chess', 'jellybeans', 'lego-gantry', 'lego-truck', 'treasure', 
          'bracelet', 'cards-small', 'eucalyptus-flowers', 'lego-bulldozer', 'lego-knights', 'stanfordbunny']
scenes = ['lego-knights']
base_dir = os.path.expanduser('~/data/3d/StanfordLF')
create_dir = base_dir.replace('StanfordLF', 'StanfordLF-colmap')
if not os.path.isdir(create_dir):
    os.mkdir(create_dir)

views = ['_00_00_', '_00_16_', '_08_08_', '_16_00_', '_16_16_']
def is_in_views(view, views=views):
    for v in views:
        if v in view:
            return True
    return False

for scene in scenes:
    ori_path = os.path.join(base_dir, scene, 'rectified')
    new_path = os.path.join(create_dir, scene)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
        print(f'{scene} copy images')
        # shutil.copytree(ori_path, os.path.join(new_path, 'images'))

        os.mkdir(os.path.join(new_path, 'images'))
        for f in os.listdir(ori_path):
            if is_in_views(f):
                shutil.copy(os.path.join(ori_path, f), os.path.join(new_path, 'images', f))
                print(f'{f} copied')

        print(f'{scene} copy finished')

    print(f'{scene} started')
    colmapGenPoses(new_path)
    print(f'{scene} finished')

