# # """
# # Generate speaker embeddings and metadata for training
# # """
# import os
# import pickle
# from model_bl import D_VECTOR
# from collections import OrderedDict
# import numpy as np
# import torch

# # C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
# #for CPU use
# C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval()
# # c_checkpoint = torch.load('3000000-BL.ckpt')
# c_checkpoint = torch.load('3000000-BL.ckpt', map_location=torch.device('cpu'))
# print(c_checkpoint.keys())
# new_state_dict = OrderedDict()
# # for key, val in c_checkpoint['model_b'].items():
# for key, val in c_checkpoint['model_b'].items():
#     new_key = key[7:]
#     new_state_dict[new_key] = val
# C.load_state_dict(new_state_dict)
# num_uttrs = 10
# len_crop = 128

# # Directory containing mel-spectrograms
# rootDir = './spmel'
# dirName, subdirList, _ = next(os.walk(rootDir))
# print('Found directory: %s' % dirName)

# print(subdirList)
# speakers = []
# for speaker in sorted(subdirList):
#     print('Processing speaker: %s' % speaker)
#     utterances = []
#     utterances.append(speaker)
#     _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    

#     # make speaker embedding
#     assert len(fileList) >= num_uttrs
#     idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
#     embs = []
#     for i in range(num_uttrs):
#         tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
#         candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
#         # choose another utterance if the current one is too short
#         while tmp.shape[0] < len_crop:
#             idx_alt = np.random.choice(candidates)
#             tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
#             candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
#         left = np.random.randint(0, tmp.shape[0]-len_crop)
#         # melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
#         melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :])
#         emb = C(melsp)
#         embs.append(emb.detach().squeeze().cpu().numpy())     
#     utterances.append(np.mean(embs, axis=0))
    
#     # create file list
#     for fileName in sorted(fileList):
#         utterances.append(os.path.join(speaker,fileName))
#     speakers.append(utterances)
    
# with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
#     pickle.dump(speakers, handle)
# print("done")

############################################
# import os
# import pickle
# from model_bl import D_VECTOR
# from collections import OrderedDict
# import numpy as np
# import torch

# C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval()
# c_checkpoint = torch.load('3000000-BL.ckpt', map_location=torch.device('cpu'))
# print(c_checkpoint.keys())
# new_state_dict = OrderedDict()
# for key, val in c_checkpoint['model_b'].items():
#     new_key = key[7:]
#     new_state_dict[new_key] = val
# C.load_state_dict(new_state_dict)
# num_uttrs = 10
# len_crop = 128

# rootDir = './spmel'
# dirName, subdirList, _ = next(os.walk(rootDir))
# print('Found directory: %s' % dirName)

# print(subdirList)
# speakers = []
# for speaker in sorted(subdirList):
#     print('Processing speaker: %s' % speaker)
#     utterances = []
#     utterances.append(speaker)
#     _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

#     assert len(fileList) >= num_uttrs
#     idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
#     embs = []
#     for i in range(num_uttrs):
#         tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]), allow_pickle=True)
#         candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
#         while tmp.shape[0] < len_crop:
#             idx_alt = np.random.choice(candidates)
#             tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]), allow_pickle=True)
#             candidates = np.delete(candidates, np.argwhere(candidates == idx_alt))
#         left = np.random.randint(0, tmp.shape[0] - len_crop)
#         melsp = torch.from_numpy(tmp[np.newaxis, left:left + len_crop, :])
#         emb = C(melsp)
#         embs.append(emb.detach().squeeze().cpu().numpy())
#     speaker_embedding = np.mean(embs, axis=0)

#     file_data = []
#     for fileName in sorted(fileList):
#         if fileName != '.DS_Store':  # Skip .DS_Store file
#             spectrogram = np.load(os.path.join(dirName, speaker, fileName), allow_pickle=True)
#             file_data.append(spectrogram)
#             # utterances.append(os.path.join(speaker,fileName))
#     # speakers.append(utterances)

#     speakers.append((utterances, speaker_embedding, file_data))

# with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
#     pickle.dump(speakers, handle)

# print("done")

# with open('./spmel/train.pkl', 'rb') as handle:
#     speakers = pickle.load(handle)

# # Print the contents of the pickle file

# # speaker_name = speaker[0]
# # embeddings = speaker[1]  # Exclude the speaker name and file paths
# # file_paths = speaker[-1]

# # print(f"Speaker: {speaker_name}")
# # print(f"Embeddings: {embeddings}")
# # print(f"File Paths: {file_paths}")
# # print()
# for speaker in speakers:
#     speaker_name = speaker[0]
#     embeddings = speaker[1]  # Exclude the speaker name and file paths
#     file_paths = speaker[2:]  # Include all file paths

#     # print(f"Speaker: {speaker_name}")
#     # print(f"Embeddings: {embeddings}")
#     # print("File Paths:")
#     for file_path in file_paths:
#         # Convert the array of numbers to a string representation
#         file_path_str = ' '.join(map(str, file_path))
#         print(file_path_str)
#     print()
################################################

import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval()
c_checkpoint = torch.load('3000000-BL.ckpt', map_location=torch.device('cpu'))
print(c_checkpoint.keys())
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

print(subdirList)
speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]), allow_pickle=True)
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]), allow_pickle=True)
            candidates = np.delete(candidates, np.argwhere(candidates == idx_alt))
        left = np.random.randint(0, tmp.shape[0] - len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left + len_crop, :])
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())
    # speaker_embedding = np.mean(embs, axis=0)
    utterances.append(np.mean(embs, axis=0))


    # file_data = []
    for fileName in sorted(fileList):
        if fileName != '.DS_Store':  # Skip .DS_Store file
            # spectrogram = np.load(os.path.join(dirName, speaker, fileName), allow_pickle=True)
            # file_data.append(spectrogram)
            utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)

    # speakers.append((utterances, speaker_embedding, file_data))

with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

print("done")

with open('./spmel/train.pkl', 'rb') as handle:
    speakers = pickle.load(handle)

# Print the contents of the pickle file

# speaker_name = speaker[0]
# embeddings = speaker[1]  # Exclude the speaker name and file paths
# file_paths = speaker[-1]

# print(f"Speaker: {speaker_name}")
# print(f"Embeddings: {embeddings}")
# print(f"File Paths: {file_paths}")
print()
for speaker in speakers:
    speaker_name = speaker[0]
    embeddings = speaker[1]  # Exclude the speaker name and file paths
    file_paths = speaker[2:]  # Include all file paths

    # print(f"Speaker: {speaker_name}")
    # print(f"Embeddings: {embeddings}")
    # print("File Paths:")
    for file_path in file_paths:
        # Convert the array of numbers to a string representation

        print(file_path)
    print()
