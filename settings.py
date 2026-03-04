from getpass import getuser
from pathlib import Path


all_paths = {'remi': {'data': Path('/home/remi/Aquineuro/Data/Moratalla/2024/data'),
                      'analysis': Path('/home/remi/Aquineuro/Data/Moratalla/2024/analysis'),
                      'figures': Path('/home/remi/Aquineuro/Data/Moratalla/2024/figures')},
             'sebas': {'data': Path('C:/Users/sebas/Desktop/Moratalla/data'),
                       'analysis': Path('C:/Users/sebas/Desktop/Moratalla/analysis'),
                       'figures': Path('C:/Users/sebas/Desktop/Moratalla/figures')
                       },
             'malek': {'data': Path(r'C:\Users\malek\Desktop\Python codes\FiberPhotometry\2024\data\Try'),
                       'analysis': Path(r'C:\Users\malek\Desktop\Python codes\FiberPhotometry\2024\data\Try\analysis'),
                       'figures': Path(r'C:\Users\malek\Desktop\Python codes\FiberPhotometry\2024\data\Try\figures')
                       }}

upaths = all_paths[getuser()]

gunmetal = '#202C39'
celadon = '#9ECE9A'
orange = '#FCAB10'
moonstone = '#2B9EB3'
colors = {'periphery': "#d1d1d1", 'maze': "#fdfcdc",
          'beige': '#fed9b7', 'center': orange, 'left': celadon, 'right': moonstone,
          'left_ring': "#cce5c9", 'right_ring': '#75cede'}
