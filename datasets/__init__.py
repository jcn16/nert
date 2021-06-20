from .blender import BlenderDataset
from .llff import LLFFDataset
from .Mydata import MyDataset
from .Mydata_srz import MyDataset_srz
from .transport import TransportDataset
from .normal import NormalDataset
dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'transport': TransportDataset,
                'normal': NormalDataset,
                'mydata': MyDataset,
                'srz':MyDataset_srz}