from wildlife_tools.features import DeepFeatures, SuperPointExtractor, SiftExtractor, DiskExtractor, AlikedExtractor
import torchvision.transforms as T
from wildlife_tools.data import ImageDataset
import pandas as pd
import os
import pickle
from utils import NewPad
import timm

parser = argparse.ArgumentParser(description="Extract global and local features.")
parser.add_argument('--metadata_csv', type=str, default='metadata.csv', help='Path to metadata CSV.')
parser.add_argument('--root_dir', type=str, default='.', help='Root directory with images.')
parser.add_argument('--output_lf', type=str, default='output_lf.pickle', help='Output path for local features.')
parser.add_argument('--output_gf', type=str, default='output_gf.pickle', help='Output path for global features.')
parser.add_argument('--ft_model', type=str, default='', help='Path to fine-tuned MegaDescriptor model (optional).')

args = parser.parse_args()

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


df = pd.read_csv(args.metadata_csv)

metadata = {'metadata':  df, 'root': args.root_dir}

dataset_lf = ImageDataset(**metadata, transform=transform_lf)
dataset_gf = ImageDataset(**metadata, transform=transform_gf)

extractor_sp = SuperPointExtractor(device = device, batch_size = BATCH_SIZE_LF)
extractor_ae = AlikedExtractor(device = device, batch_size = BATCH_SIZE_LF)
extractor_disk = DiskExtractor(device = device, batch_size = BATCH_SIZE_LF)

output_sp = extractor_sp(dataset_lf)
output_ae = extractor_ae(dataset_lf)
output_disk = extractor_disk(dataset_lf)

with open(args.output_lf, 'wb') as f:
    pickle.dump([output_sp, output_ae, output_disk], f)

backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', num_classes=0, pretrained=True)
if args.ft_model:
    state_dict = torch.load(args.ft_model)
    backbone.load_state_dict(state_dict)

extractor_gf = DeepFeatures(backbone, batch_size=BATCH_SIZE_GF, device=device)
output_gf = extractor_gf(dataset_gf)

with open(args.output_gf, 'wb') as f:
    pickle.dump(output_gf, f)
