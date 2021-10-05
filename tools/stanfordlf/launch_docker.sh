# docker build -t lsongx/colmap:run .
docker pull lsongx/colmap:run
# nvidia-docker run -it --rm colmap bash
docker run -it --rm --gpus=all --mount type=bind,source="$(pwd)/../..",target=/workspace --mount type=bind,source=/home/uss00032/data/3d,target=/root/data/3d lsongx/colmap:run /bin/bash
# python3

# docker push lsongx/colmap:run
