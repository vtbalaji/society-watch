#!/bin/sh
# Fetch the public test clips used during development into samples/.
set -e
cd "$(dirname "$0")/../samples"
UA="incident-detector-test/0.1"
curl -sL -A "$UA" -o dogs-resting-grass.ogv "https://upload.wikimedia.org/wikipedia/commons/3/38/Africanis_Dogs_Resting_In_Grass.ogv"
curl -sL -A "$UA" -o dog-park.webm "https://upload.wikimedia.org/wikipedia/commons/9/97/Atago_Dog_Park_Grand_Opening%2C_MCAS_Iwakuni%2C_Japan%2C_2026_%28B-Roll%29_%281008314%29.webm"
for f in dogs-resting-grass.ogv dog-park.webm; do
  ffmpeg -hide_banner -loglevel error -y -i "$f" -an -c:v h264_videotoolbox -b:v 6M "${f%.*}.mp4" && rm "$f"
done
../.venv/bin/python -c "from supervision.assets import download_assets, VideoAssets as V; download_assets(V.PEOPLE_WALKING)"
