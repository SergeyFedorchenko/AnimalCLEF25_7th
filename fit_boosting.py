import pickle
import torch.nn.functional as F
import numpy as np
import pickle
import os
import pandas as pd
import torch
import lightgbm as lgb

parser = argparse.ArgumentParser(description="Fit LightGBM on pairwise similarity features and generate submission.")
parser.add_argument('--metadata_csv', type=str, default='metadata.csv', help='Path to metadata CSV.')
parser.add_argument('--output_gf', type=str, default='output_gf.pickle', help='Path to global features.')
parser.add_argument('--output_kp', type=str, default='output_kp.pickle', help='Path to keypoint match features.')
parser.add_argument('--submission_csv', type=str, default='submission.csv', help='Output path for submission file.')
parser.add_argument('--best_k', type=int, default=150, help='Top-K most similar pairs per query.')
parser.add_argument('--thresh', type=float, default=0.75, help='Threshold below which predictions are labeled "new_individual".')

args = parser.parse_args()


with open(args.output_gf, 'rb') as f:
    output_gf = pickle.load(f)

with open(args.output_kp, 'rb') as f:
    output_sp, output_ae, output_disk = pickle.load(f)

device = 'cuda'
BEST_K = args.best_k
TH = args.thresh

current_file_path = os.getcwd()
df = pd.read_csv(args.metadata_csv)
df['orient_class'] = df['orientation'].fillna('no').map({x:i for i,x in enumerate(sorted(df['orientation'].fillna('no').unique()))})

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

cos_sim_chunk_pairs = np.array(cos_sim_chunk_pairs)

id1 = val_data_idx[[x[0] for x in cos_sim_chunk_pairs]]
id2 = train_data_idx[[x[1] for x in cos_sim_chunk_pairs]]

df_lgb = pd.DataFrame()
df_lgb['id1'] = id1
df_lgb['id2'] = id2
df_lgb['score_mega'] = cos_sim_chunk_values_train.cpu().numpy()[:, :150].flatten()
df_lgb['target1'] = df['identity'].values[id1]
df_lgb['target2'] = df['identity'].values[id2]
df_lgb['orient1'] = df['orient_class'].values[id1]
df_lgb['orient2'] = df['orient_class'].values[id2]
df_lgb['species1'] = df['species'].values[id1]
df_lgb['species2'] = df['species'].values[id2]
df_lgb['score_sp_5'] = [(x['scores'] > 0.5).sum() for x in output_sp]
df_lgb['score_sp_8'] = [(x['scores'] > 0.8).sum() for x in output_sp]
df_lgb['score_ds_5'] = [(x['scores'] > 0.5).sum() for x in output_disk]
df_lgb['score_ds_8'] = [(x['scores'] > 0.8).sum() for x in output_disk]
df_lgb['score_ae_5'] = [(x['scores'] > 0.5).sum() for x in output_ae]
df_lgb['score_ae_8'] = [(x['scores'] > 0.8).sum() for x in output_ae]
df_lgb = df_lgb[df_lgb['id1'] != df_lgb['id2']].reset_index(drop = True)
df_lgb['target'] = (df_lgb['target1'] == df_lgb['target2']).astype('int')

param_lgb = {
'boost': 'gbdt',
'learning_rate': 0.1,
'num_leaves': 32,
'objective': 'binary',
'verbosity':-1
}

train_cols = ['orient1', 'orient2', 'score_mega',
       'score_sp_5', 'score_sp_8', 'score_ds_5',
       'score_ds_8', 'score_ae_5', 'score_ae_8', ]

train_df_lgb = df_lgb[~df_lgb['target1'].isnull()]
test_df_lgb = df_lgb[df_lgb['target1'].isnull()].copy()
lgb_tr = lgb.Dataset(np.array(train_df_lgb[train_cols]), np.array(train_df_lgb['target']))

bst = lgb.train(param_lgb, lgb_tr, num_boost_round=50)

test_df_lgb['preds'] = bst.predict(test_df_lgb[train_cols])
preds_array = test_df_lgb['preds'].values.reshape((test_df_lgb.shape[0] // BEST_K, BEST_K))
ids_array = test_df_lgb['id2'].values.reshape((test_df_lgb.shape[0] // BEST_K, BEST_K))

argsort = np.argsort(preds_array, axis = 1)[:, -1]
preds_final = []
for i, x in enumerate(ids_array):
    preds_final += [ (preds_array[i, argsort[i]], ids_array[i, argsort[i]]) ]

preds_animal = [df['identity'].values[x[1]] for x in preds_final]

preds_animal = np.array([df['identity'].values[x[1]] for x in preds_final])
preds_animal[np.array([x[0] for x in preds_final]) < TH] = 'new_individual'

submission = pd.DataFrame()
submission['image_id'] = test_df_lgb['id1'][::BEST_K].values
submission['identity'] = preds_animal
os.makedirs(os.path.dirname(args.submission_csv), exist_ok=True)
submission.to_csv(args.submission_csv, index = None)
