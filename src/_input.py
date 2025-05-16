import pandas as pd
import numpy as np
from itertools import product
import math

# Read CSV
def extract_features(file_name):
  df = pd.read_csv(file_name)
  sequences = df['sequence'] if 'sequence' in df.columns else df['sequences']

# Allowed tokens
  NUCS = ['A', 'U', 'G', 'C', 'X']
  TOKEN_IDX = {n: i for i, n in enumerate(NUCS)}

  def one_hot_encode(seq):
      vec = np.zeros((len(seq), len(NUCS)))
      for i, n in enumerate(seq):
          vec[i, TOKEN_IDX[n]] = 1
      return vec

  def sinusoidal_pos_encoding(length, d_model):
      pos = np.arange(length)[:, np.newaxis]
      i = np.arange(d_model)[np.newaxis, :]
      angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
      angle_rads = pos * angle_rates
      sines = np.sin(angle_rads[:, 0::2])
      cosines = np.cos(angle_rads[:, 1::2])
      return np.concatenate([sines, cosines], axis=-1)

  def can_pair(a, b):
      return (a, b) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')]

  def nussinov_fold(seq):
      n = len(seq)
      dp = np.zeros((n, n), dtype=int)
      bt = [[[] for _ in range(n)] for _ in range(n)]

      for l in range(1, n):
          for i in range(n - l):
              j = i + l
              if j - i >= 4:
                  max_val = dp[i+1][j]
                  bt[i][j] = bt[i+1][j]
                  if dp[i][j-1] > max_val:
                      max_val = dp[i][j-1]
                      bt[i][j] = bt[i][j-1]
                  if can_pair(seq[i], seq[j]):
                      paired = dp[i+1][j-1] + 1
                      if paired > max_val:
                          max_val = paired
                          bt[i][j] = bt[i+1][j-1] + [(i, j)]
                  for k in range(i+1, j):
                      comb = dp[i][k] + dp[k+1][j]
                      if comb > max_val:
                          max_val = comb
                          bt[i][j] = bt[i][k] + bt[k+1][j]
                  dp[i][j] = max_val
      structure = ['.'] * n
      for i, j in bt[0][n-1]:
          structure[i] = '('
          structure[j] = ')'
      return ''.join(structure), bt[0][n-1]

  def boltzmann_pair_matrix(seq, pairs, temperature=300):
      n = len(seq)
      kB = 0.001987 # kcal/mol·K
      beta = 1 / (kB * temperature)
      P = np.zeros((n, n))
      for (i, j) in pairs:
          energy = -2.0 if (seq[i], seq[j]) in [('G','C'), ('C','G')] else -1.0
          P[i][j] = math.exp(-beta * energy)
      Z = np.sum(P)
      if Z > 0:
          P /= Z
      return P

  def compute_entropy(P):
      eps = 1e-10
      entropy = -np.nansum(P * np.log(P + eps))
      return entropy


# Final dataframe
  records = []

  for seq in sequences:
      seq = seq.strip().upper()
      token_embed = one_hot_encode(seq)
      pos_embed = sinusoidal_pos_encoding(len(seq), len(NUCS))
      structure, pairs = nussinov_fold(seq)
      bp_matrix = boltzmann_pair_matrix(seq, pairs)
      entropy = compute_entropy(bp_matrix)
      records.append({
          "sequence": seq,
          "token_embeddings": token_embed,
          "positional_embeddings": pos_embed,
          "pair_list": pairs,
          "dot_bracket": structure,
          "base_pair_prob_matrix": bp_matrix,
          "sequence_length": len(seq),
          "entropy": entropy
      })

# Convert to DataFrame
  final_df = pd.DataFrame(records)
