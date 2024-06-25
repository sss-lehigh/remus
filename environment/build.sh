architecture=$(uname -m)
BUILDER="builder"

# Check for the existance of the builder context
if docker buildx ls | grep -q $BUILDER; then
    echo "Builder context exists."
else
    echo "Builder context does not exist. Creating..."
    docker buildx create --name $BUILDER --use
fi

# Inspecting build context for cross compilation
docker buildx inspect --bootstrap

# Ensure binfmt is installed
docker run --rm tonistiigi/binfmt --install all

echo "Building for $architecture..."
if [[ "$architecture" == "arm64" ]]; then
# We use buildx to build for arm64
    docker buildx build --no-cache --platform linux/amd64 -t remus .
elif [[ "$architecture" == "x86_64" ]]; then 
    docker build --no-cache -t remus .
else
    echo "Unknown architecture $architecture."
fi

