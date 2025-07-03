import numpy as np
from wildlife_tools.similarity.pairwise.collectors import CollectAll
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
import pickle
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import torch

parser = argparse.ArgumentParser(description="Generate keypoint matches for candidate pairs.")
parser.add_argument('--metadata_csv', type=str, default='metadata.csv', help='Path to metadata CSV.')
parser.add_argument('--output_gf', type=str, default='output_gf.pickle', help='Path to global features pickle.')
parser.add_argument('--output_lf', type=str, default='output_lf.pickle', help='Path to local features pickle.')
parser.add_argument('--output_kp', type=str, default='output_kp.pickle', help='Output path for keypoint match results.')
parser.add_argument('--best_k', type=int, default=150, help='Top-K most similar pairs per query.')
parser.add_argument('--batch_size_match', type=int, default=128, help='Batch size for LightGlue matchers.')
args = parser.parse_args()

with open(args.output_lf, 'rb') as f:
    output_sp, output_ae, output_disk = pickle.load(f)

with open(args.output_gf, 'rb') as f:
    output_gf = pickle.load(f)

device = 'cuda'
BEST_K = args.best_k
BATCH_SIZE_MATCH = args.batch_size_match

current_file_path = os.getcwd()
df = pd.read_csv(args.metadata_csv)

train_data_idx = np.where(df['split'] == 'database')[0]
val_data_idx =  np.concatenate((np.where(df['split'] == 'database')[0][::10], 
                            np.where(df['split'] == 'query')[0]))

embeds_val = torch.from_numpy(np.array([output_gf[i][0] for i in val_data_idx])).to(device)
embeds_val = F.normalize(embeds_val, dim=1)

embeds_train = torch.from_numpy(np.array([output_gf[i][0] for i in train_data_idx])).to(device)
embeds_train = F.normalize(embeds_train, dim=1)

cos_sim_chunk_values_train = torch.mm(embeds_val, embeds_train.transpose(0, 1))
cos_sim_chunk_values_train, cos_sim_chunk_train = cos_sim_chunk_values_train.sort(dim=1, descending=True)

cos_sim_chunk_pairs = []
for i, val in enumerate(cos_sim_chunk_train.cpu().detach().numpy()):
    for x in val[:BEST_K]:
        cos_sim_chunk_pairs += [(i,x)]

collector = CollectAll()

matcher_sp = MatchLightGlue(features='superpoint', collector=collector, device = device, batch_size = BATCH_SIZE_MATCH)
matcher_ae = MatchLightGlue(features='aliked', collector=collector, device = device, batch_size = BATCH_SIZE_MATCH)
matcher_disk = MatchLightGlue(features='disk', collector=collector, device = device, batch_size = BATCH_SIZE_MATCH)

features_query = [output_sp[x] for x in val_data_idx]
features_database = [output_sp[x] for x in train_data_idx]
output_sp = matcher_sp(features_query, features_database,  pairs=cos_sim_chunk_pairs)

features_query = [output_ae[x] for x in val_data_idx]
features_database = [output_ae[x] for x in train_data_idx]
output_ae = matcher_ae(features_query, features_database,  pairs=cos_sim_chunk_pairs)

features_query = [output_disk[x] for x in val_data_idx]
features_database = [output_disk[x] for x in train_data_idx]
output_disk = matcher_disk(features_query, features_database,  pairs=cos_sim_chunk_pairs)

os.makedirs(os.path.dirname(args.output_kp), exist_ok=True)
with open(args.output_kp, 'wb') as f:
    pickle.dump([output_sp, output_ae, output_disk], f)
