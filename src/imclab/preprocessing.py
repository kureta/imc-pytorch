from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm


def _prepare_single(file_path):
    sound = librosa.load(file_path, sr=44100, mono=True)[0]
    stft = librosa.stft(sound, n_fft=1024, hop_length=512, window='hann')
    spec = np.abs(stft)[1:]
    db = librosa.amplitude_to_db(spec)

    return db


def prepare(data_dir):
    pd_name = data_dir.stem.lower().replace(" ", "_") + ".npy"
    pd_path = Path.cwd() / 'data' / pd_name
    if pd_path.exists():
        print('Loading processed data')
        return np.load(pd_path)

    print('Processing data')
    all_files = list(data_dir.glob("*.mp3"))
    size = len(all_files)
    with Pool(12) as p:
        spectra = list(tqdm(p.imap_unordered(_prepare_single, all_files), total=size))

    print('Saving processed data')
    data = np.concatenate(spectra, axis=1).T
    np.save(pd_path, data)

    return data


def image_preprocess(pred: torch.Tensor) -> np.array:
    image = pred.T.flip(0).unsqueeze(0)
    image = image - image.min()
    image = image / image.max()

    return image


def audio_preprocess(pred: torch.Tensor) -> np.array:
    result = np.zeros((513, pred.shape[0]))
    result[1:, :] = pred.T.cpu().numpy() * 13. - 36.
    result = result
    audio = librosa.griffinlim(librosa.db_to_amplitude(result), n_iter=100, hop_length=512, n_fft=1024, window='hann')

    return audio
