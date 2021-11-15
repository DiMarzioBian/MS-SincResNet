import os
import pandas as pd
import pickle
import zipfile

track_list = []
filename_list = []
discard_list = ['098565.mp3', '098567.mp3', '098569.mp3', '099134.mp3', '108925.mp3', '133297.mp3',
                'checksums', 'README.txt']

# Unzipping
print('Unzipping fma_small.zip')
zip_file = zipfile.ZipFile('fma_small.zip')
for name in zip_file.namelist():
    zip_file.extract(name)
zip_file.close()

print('Unzipping fma_metadata.zip')
zip_file = zipfile.ZipFile('fma_metadata.zip')
for name in zip_file.namelist():
    zip_file.extract(name)
zip_file.close()

# Traverse all files
for dir_name, _, file_list in os.walk('fma_small'):
    for track_name in file_list:
        if track_name in discard_list:
            continue
        track_list.append(int(track_name[:-4]))
        filename_list.append(dir_name[-3:] + '/' + track_name)

# Discard unused columns and rows
df_track_full = pd.read_csv('fma_metadata/tracks.csv', skiprows=[0])
df_track_full = df_track_full[['Unnamed: 0', 'genre_top']].iloc[1:]
df_track_full['track_id'] = df_track_full['Unnamed: 0'].astype(int)

df_track = df_track_full[df_track_full.track_id.isin(track_list)][['track_id', 'genre_top']]

genre_mapper = {}
for i, lbl in enumerate(df_track['genre_top'].unique()):
    genre_mapper[lbl] = i

df_track['genre_top'] = [genre_mapper[x] for x in df_track['genre_top']]
df_track = df_track.reset_index(drop=True)
df_track.sort_values('track_id', inplace=True)

# Make dict
dict_genre = {}
for fn, x in zip(filename_list, df_track.iterrows()):
    dict_genre[fn] = x[1][1]

file = open('track_genre.pkl', 'wb')
pickle.dump(dict_genre, file)

print('\n[info] Finished creating pkl file.\n')
