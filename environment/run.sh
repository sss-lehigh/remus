architecture=$(uname -m)

echo "Running for $architecture..."
if [[ "$architecture" == "arm64" ]]; then
    docker run --privileged --rm --platform linux/amd64 --env-file .env -v $(realpath ..):/root --name remus -it remus
elif [[ "$architecture" == "x86_64" ]]; then 
    docker run --privileged --rm --env-file .env -v $(realpath ..):/root --name remus -it remus
else
    echo "Unknown architecture $architecture."
fi