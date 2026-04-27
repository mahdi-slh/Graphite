
# Running Fat
For running fat dataset should be downloaded and put in the root folder as 'fat'
the link  to download the dataset¦:
https://research.nvidia.com/publication/2018-06_Falling-Things

The dataset is then to be parsed using parse.py, a folder with the name data should be created and the parse.py parses all the object folders inside that

# Docker
A Dockerfile has been added, the image is also available on Dockerhub @shervn.

## HOW TO BUILD and RUN

### BUILD
If you pull the image, you can skip this part.
`docker build -t tag .`

### RUN
Run the following code:
`docker run --rm -it --init  --ipc=host --runtime=nvidia {if Cuda}   --user="$(id -u):$(id -g)"   --volume="$PWD:/app" -e NVIDIA_VISIBLE_DEVICES=0 shervn/idp python3 {PATH_TO_PYTHON_CODE}`

