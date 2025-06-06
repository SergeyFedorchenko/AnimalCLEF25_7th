from wildlife_tools.features import DeepFeatures, SuperPointExtractor, SiftExtractor, DiskExtractor, AlikedExtractor
import torchvision.transforms as T
from wildlife_tools.data import ImageDataset
import pandas as pd
import os
import pickle
from utils import NewPad
import timm

transform_lf = T.Compose([
    NewPad(),
    T.Resize([512, 512]),
    T.ToTensor(),
])

transform_gf = T.Compose([
    NewPad(),
    T.Resize([384, 384]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

BATCH_SIZE_LF = 128
BATCH_SIZE_GF = 32
device = 'cuda'

# path_to_ft_model = '/content/ft_model.pt'
path_to_ft_model = ''
df = pd.read_csv(f'metadata.csv')
current_file_path = os.getcwd()

metadata = {'metadata':  df, 'root': current_file_path}

dataset_lf = ImageDataset(**metadata, transform=transform_lf)
dataset_gf = ImageDataset(**metadata, transform=transform_gf)

extractor_sp = SuperPointExtractor(device = device, batch_size = BATCH_SIZE_LF)
extractor_ae = AlikedExtractor(device = device, batch_size = BATCH_SIZE_LF)
extractor_disk = DiskExtractor(device = device, batch_size = BATCH_SIZE_LF)

output_sp = extractor_sp(dataset_lf)
output_ae = extractor_ae(dataset_lf)
output_disk = extractor_disk(dataset_lf)

with open('output_lf.pickle', 'wb') as f:
    pickle.dump([output_sp, output_ae, output_disk], f)

backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', num_classes=0, pretrained=True)
if path_to_ft_model:
    state_dict = torch.load(path_to_ft_model)
    backbone.load_state_dict(state_dict)

extractor_gf = DeepFeatures(backbone, batch_size=BATCH_SIZE_GF, device=device)
output_gf = extractor_gf(dataset_gf)

with open('output_gf.pickle', 'wb') as f:
    pickle.dump(output_gf, f)