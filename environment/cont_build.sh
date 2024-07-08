#!/usr/bin/env bash
GREEN=$(tput setaf 2)
NC=$(tput sgr0)

architecture=$(uname -m)
BUILDER="builder"

check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    else 
        echo "${GREEN}$1 successful.${NC}"
    fi
}

echo "Building for $architecture..."
if [[ "$architecture" == "arm64" ]]; then
    echo "IM NOT WORKING"
    # Check for the existance of the builder context
    if docker buildx ls | grep -q $BUILDER; then
        echo "Builder context already exists."
        docker buildx rm $BUILDER
        check_success "docker buildx rm"
        echo "Rebuilding..."
    else 
        echo "Builder context does not yet exist."
    fi
    # Regardless we are building a fresh context
    docker buildx create --name $BUILDER --use
    check_success "docker buildx create"

    # Inspecting build context for cross compilation
    docker buildx inspect --bootstrap
    check_success "docker buildx inspect"

    # Ensure binfmt is installed
    docker run --rm --privileged tonistiigi/binfmt --install all
    check_success "installing binfmt"
    # We use buildx to build for arm64
    docker buildx build --no-cache --platform linux/amd64 -t remus:latest --load .
elif [[ "$architecture" == "x86_64" ]]; then 
    docker build --no-cache -t remus:latest .
else
    echo "Unknown architecture $architecture."
fi

