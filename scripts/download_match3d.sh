#!/usr/bin/env bash
# Download the 3DMatch Geometric Registration Benchmark test split.
# Lays out files under $MATCH3D_DIR (defaults to data/match3d/) using the
# short scene names that Graphite's `Match3DDataset` expects:
#     home1, home2, hotel1, hotel2, hotel3, kitchen, mit, study
#
# Usage:
#     ./scripts/download_match3d.sh                 # downloads everything (~300 MB)
#     SCENES="kitchen home1" ./scripts/download_match3d.sh   # subset only
set -euo pipefail

MATCH3D_DIR=${MATCH3D_DIR:-"$(cd "$(dirname "$0")/.." && pwd)/data/match3d/"}
mkdir -p "$MATCH3D_DIR/fragments" "$MATCH3D_DIR/eval_files"

BASE_URL="http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments"

# Mapping: <short-name>::<full-3dmatch-name>
DEFAULT_SCENES=(
    "home1::sun3d-home_at-home_at_scan1_2013_jan_1"
    "home2::sun3d-home_md-home_md_scan9_2012_sep_30"
    "hotel1::sun3d-hotel_uc-scan3"
    "hotel2::sun3d-hotel_umd-maryland_hotel1"
    "hotel3::sun3d-hotel_umd-maryland_hotel3"
    "kitchen::7-scenes-redkitchen"
    "mit::sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"
    "study::sun3d-mit_76_studyroom-76-1studyroom2"
)

REQUESTED=${SCENES:-}

for entry in "${DEFAULT_SCENES[@]}"; do
    short="${entry%%::*}"
    full="${entry##*::}"

    if [[ -n "$REQUESTED" && ! " $REQUESTED " =~ \ $short\  ]]; then
        continue
    fi

    frag_target="$MATCH3D_DIR/fragments/$short"
    eval_target="$MATCH3D_DIR/eval_files/$short"

    if [[ ! -d "$frag_target" || -z "$(ls -A "$frag_target" 2>/dev/null)" ]]; then
        echo "[fragments] $short  ($full)"
        tmp_zip="$(mktemp --suffix=.zip)"
        curl -sSfL -o "$tmp_zip" "$BASE_URL/$full.zip"
        unzip -q -o "$tmp_zip" -d "$MATCH3D_DIR/fragments/"
        rm -f "$tmp_zip"
        # rename the unzipped folder to the short name
        rm -rf "$frag_target"
        mv "$MATCH3D_DIR/fragments/$full" "$frag_target"
    else
        echo "[fragments] $short  already present, skipping download"
    fi

    if [[ ! -f "$eval_target/gt.log" ]]; then
        echo "[eval]      $short"
        tmp_zip="$(mktemp --suffix=.zip)"
        curl -sSfL -o "$tmp_zip" "$BASE_URL/$full-evaluation.zip"
        unzip -q -o "$tmp_zip" -d "$MATCH3D_DIR/eval_files/"
        rm -f "$tmp_zip"
        rm -rf "$eval_target"
        mv "$MATCH3D_DIR/eval_files/$full-evaluation" "$eval_target"
    else
        echo "[eval]      $short  already present, skipping download"
    fi
done

echo
echo "3DMatch test data ready at: $MATCH3D_DIR"
echo "  fragments/<scene>/cloud_bin_X.ply"
echo "  eval_files/<scene>/gt.log"
