#!/usr/bin/env bash
# Copy plots from a given experiment into assets/plots for README display.
# Usage: bash scripts/prepare_readme_assets.sh <experiment_name>
set -euo pipefail

EXP="${1:-example_experiment}"
SRC="data/results/${EXP}/plots"
DST="assets/plots"

# Create destination directory if it doesn't exist
mkdir -p "${DST}"

copy_plot() {
  local base="$1"
  if [ -f "${SRC}/${base}.png" ]; then
    cp "${SRC}/${base}.png" "${DST}/${base}.png"
    echo "Copied ${base}.png"
  else
    echo "Warning: ${SRC}/${base}.png not found; leaving placeholder."
  fi
}

echo "Copying plots from experiment '${EXP}' to README assets..."

copy_plot "sample_molecules"
copy_plot "qed_distribution"
copy_plot "molwt_distribution"
copy_plot "logp_distribution"
copy_plot "chemical_space_pca"
copy_plot "generator_pretraining_loss"
copy_plot "discriminator_pretraining_loss"
copy_plot "gan_training_history"

echo "Done. Update README image extensions if you switch to PNGs."
echo "Note: This script copies PNG files but README currently uses SVG placeholders."
echo "You may want to update the README to use PNG extensions after running experiments."