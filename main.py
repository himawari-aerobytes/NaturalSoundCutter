import librosa
import numpy as np
from pydub import AudioSegment

# 音声ファイルのパス
input_file = 'input.mp3'
output_file = 'summary_output.mp3'

# 目標の長さ（秒）
target_duration = 60.0  # 60秒

# 音声ファイルを読み込む
y, sr = librosa.load(input_file, sr=None)

# 曲の全長
total_duration = librosa.get_duration(y=y, sr=sr)

# メル周波数ケプストラム係数（MFCC）を計算
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# セグメンテーション（曲を区間に分割）
import scipy
from sklearn.cluster import KMeans

# フレームごとの特徴量を取得
S = np.abs(librosa.stft(y))
S_db = librosa.amplitude_to_db(S, ref=np.max)

# 特徴量の次元削減
import sklearn
pca = sklearn.decomposition.PCA(n_components=2)
S_pca = pca.fit_transform(S_db.T)

# KMeansでクラスタリング（例として4つのセグメントに分割）
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(S_pca)
labels = kmeans.labels_

# 各セグメントの開始フレームと終了フレームを取得
boundaries = np.nonzero(np.diff(labels))[0]
segment_frames = np.concatenate(([0], boundaries, [len(labels)-1]))

# 各セグメントの開始時間と終了時間を取得
segment_times = librosa.frames_to_time(segment_frames, sr=sr)

# セグメントごとのエネルギーを計算
segment_energies = []
for i in range(len(segment_times)-1):
    start_time = segment_times[i]
    end_time = segment_times[i+1]
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = y[start_sample:end_sample]
    energy = np.sum(segment**2)
    segment_energies.append((energy, start_sample, end_sample))

# エネルギーの高い順にセグメントをソート
segment_energies.sort(reverse=True)

# 選択するセグメントを決定
selected_segments = []
current_duration = 0.0
for energy, start_sample, end_sample in segment_energies:
    segment_duration = (end_sample - start_sample) / sr
    if current_duration + segment_duration <= target_duration:
        selected_segments.append((start_sample, end_sample))
        current_duration += segment_duration
    if current_duration >= target_duration:
        break

# 選択したセグメントを時間順にソート
selected_segments.sort(key=lambda x: x[0])

# 選択したセグメントをつなぎ合わせる
y_summary = np.array([], dtype=y.dtype)
for start_sample, end_sample in selected_segments:
    y_summary = np.concatenate((y_summary, y[start_sample:end_sample]))

# フェードイン・フェードアウトの設定（秒）
fade_in_duration = 2.0
fade_out_duration = 2.0

# サンプル数に変換
fade_in_samples = int(fade_in_duration * sr)
fade_out_samples = int(fade_out_duration * sr)

# フェードインを適用
if len(y_summary) > fade_in_samples:
    fade_in_curve = np.linspace(0, 1, fade_in_samples)
    y_summary[:fade_in_samples] *= fade_in_curve

# フェードアウトを適用
if len(y_summary) > fade_out_samples:
    fade_out_curve = np.linspace(1, 0, fade_out_samples)
    y_summary[-fade_out_samples:] *= fade_out_curve

# NumPy配列をAudioSegmentに変換
y_int16 = np.int16(y_summary / np.max(np.abs(y_summary)) * 32767)
audio = AudioSegment(
    y_int16.tobytes(),
    frame_rate=sr,
    sample_width=2,  # 16ビット=2バイト
    channels=1
)

# 処理した音声をMP3としてエクスポート
audio.export(output_file, format='mp3')
