docker build -t colmap .
# nvidia-docker run -it --rm colmap bash
docker run -it --rm --gpus=all --mount type=bind,source="$(pwd)/../..",target=/workspace --mount type=bind,source=/home/uss00032/data/3d,target=/home/user/data/3d --mount type=bind,source=`pwd`/../../data/out,target=/app/data/out colmap /bin/bash
# python3