#!/bin/bash
# =============================================================================
# nemotron-1d — HyenaND vs Mamba-2 vs Gated Delta Product at NEMOTRON's config.
#
# A different question from the reach/flash sweeps, and deliberately a different
# setup. Those ask "how far does each operator scale" at a tiny width out to 16.7M
# tokens. This one asks the only question Nemotron's architecture committee will
# care about: **at the config and sequence length Nemotron actually pretrains at,
# how does HyenaND compare to the Mamba-2 it would replace, and to the Gated Delta
# Product that is the competing replacement?**
#
# Why the reach sweep does not answer it: Nemotron pretrains at L=8192, and per
# nemotron_workspace/FLOPS_MATCHING.md the long conv is ~3.4% of a Hyena layer
# there — "the layer is projection-bound". A 6496x advantage at 16.7M says nothing
# about 8192. The comparison here is projections, not asymptotics.
#
# Config — the 1B A315M-88B rung, the decision gate in NEMOTRON_HYENA_PLAN.md:
#   hidden 768 | mamba_state_dim 128 | mamba_head_dim 64 | mamba_num_groups 8
#   mamba_num_heads 24 (= d_inner 1536 / 64) | expand 2 | L = 8192 | bf16
# Sources: NEMOTRON_HYENA_PLAN.md:220,248 and FLOPS_MATCHING.md:125.
#
# GDP reads the SAME mamba_* fields as Mamba-2 and hardcodes num_householder in the
# source (3 in gated_delta_product_original_v4.py, 2 in _nh2), so sizing all three
# from one config is exactly what a drop-in `--spec` swap produces. Its in_proj is
# consequently wider: 10336 vs Mamba-2's 5144 at M=3 (2.01x) — an arithmetic fact
# about GEMM width. Whether that becomes 2x *time* is what this benchmark measures.
#
# Mamba runs UNIDIRECTIONAL here (--mamba-causal). The other sweeps use
# bidirectional, which suits vision but is wrong for a language model.
#
# !! Hyena is NOT FLOP-matched here. `_hyena_mixer_cfg` hardcodes a 3*hidden_dim
# mixer with no expansion knob, i.e. evo2's default e = 1, which FLOPS_MATCHING.md
# measures at **0.423x a Mamba-2 layer**. So the hyena column is a materially
# SMALLER layer than the mamba and gdp columns, and its timings are not a like-for-
# like comparison against them. FLOPS_MATCHING.md derives e = 2.36 for parity;
# implementing it needs an expansion parameter threaded through _hyena_mixer_cfg
# (in_proj h -> 3*e*h, long conv on e*h channels, out_proj e*h -> h). Until then,
# read hyena-vs-mamba/gdp as indicative only.
#
# The gdp-vs-mamba pairing IS like-for-like: both are sized from the same mamba_*
# config, which is exactly what a drop-in --spec swap produces.
#
# Usage:
#   scripts/slurm/submit_nemotron_1d.sh                    # 1B rung, L sweep
#   HIDDEN_DIM=2048 MAMBA_NUM_HEADS=64 scripts/slurm/submit_nemotron_1d.sh   # 14B rung
#   GDP_HOUSEHOLDER=2 scripts/slurm/submit_nemotron_1d.sh  # the _nh2 variant
#   SEQ_LENS="8192" scripts/slurm/submit_nemotron_1d.sh    # operating point only
#
# Writes benchmarks/results/nemotron_1d.{jsonl,png,pdf} — a NEW output stem, so the
# reach/flash results are untouched.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── The 1B A315M rung ────────────────────────────────────────────────────────
HIDDEN_DIM="${HIDDEN_DIM:-768}"
MAMBA_STATE_DIM="${MAMBA_STATE_DIM:-128}"
MAMBA_HEADDIM="${MAMBA_HEADDIM:-64}"
MAMBA_NGROUPS="${MAMBA_NGROUPS:-8}"
MAMBA_EXPAND="${MAMBA_EXPAND:-2}"
GDP_HOUSEHOLDER="${GDP_HOUSEHOLDER:-3}"

# Sweep around Nemotron's 8192, wide enough to show both the operating point and
# where each operator walls. L = R in 1D.
SEQ_LENS="${SEQ_LENS:-1024 2048 4096 8192 16384 32768 65536 131072 262144}"

MIXERS="${MIXERS:-hyena mamba gdp}"

# Attention needs hidden/num_heads to divide by 2 for 1D RoPE; 768/12 = 64.
NUM_HEADS="${NUM_HEADS:-12}"

DATA_DIM=1
OUT="${OUT:-nemotron_1d}"
FFT_BACKEND="${FFT_BACKEND:-subq_ops}"   # 1D causal fused long conv
SHORT_CONV="${SHORT_CONV:-subq_ops}"     # fused causal short conv

echo "[nemotron-1d] hidden=${HIDDEN_DIM} state_dim=${MAMBA_STATE_DIM} head_dim=${MAMBA_HEADDIM}"
echo "[nemotron-1d] ngroups=${MAMBA_NGROUPS} expand=${MAMBA_EXPAND} gdp_M=${GDP_HOUSEHOLDER}"
echo "[nemotron-1d] mixers='${MIXERS}'  L='${SEQ_LENS}'  (Nemotron pretrains at 8192)"

export DATA_DIM MIXERS HIDDEN_DIM NUM_HEADS MAMBA_HEADDIM MAMBA_EXPAND OUT FFT_BACKEND SHORT_CONV
export MAMBA_STATE_DIM MAMBA_NGROUPS GDP_HOUSEHOLDER
export RESOLUTIONS="${SEQ_LENS}"
export EXTRA_ARGS="--mamba-state-dim ${MAMBA_STATE_DIM} --mamba-ngroups ${MAMBA_NGROUPS} --mamba-causal --gdp-householder ${GDP_HOUSEHOLDER}"
export NEEDS_FLA=1

exec sbatch \
    --job-name="nvsubq-nemotron-1d" \
    --time="${TIME_LIMIT:-02:00:00}" \
    --export=ALL \
    "${HERE}/submit_forward_time_nd.sh"
