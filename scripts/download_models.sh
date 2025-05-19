#!/bin/bash

# StormSight Model Downloader
# --------------------------
# Downloads required models from different sources
# Usage: bash scripts/download_models.sh

set -euo pipefail

# Configuration
MODEL_DIR="models"
declare -A MODEL_URLS=(
    ["deraining/deraining.pth"]="https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth"
    ["detection/craft_mlt_25k.pth"]="https://drive.usercontent.google.com/u/0/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ&export=download"
)

declare -A MODEL_SHA256=(
    ["deraining/deraining.pth"]="225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681" 
    ["detection/craft_mlt_25k.pth"]="4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17" 
)

# Create directory structure
mkdir -p "${MODEL_DIR}/deraining"
mkdir -p "${MODEL_DIR}/detection"

download_file() {
    local url=$1
    local dest=$2
    
    echo "Downloading ${dest} from ${url}"
    
    if [[ $url == *"google.com"* ]]; then
        # Google Drive download with cookie handling
        local cookie_file=$(mktemp)
        local confirm=$(wget --quiet --save-cookies "${cookie_file}" --keep-session-cookies \
                           --no-check-certificate "${url}" -O- | \
                           sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
        
        wget --load-cookies "${cookie_file}" \
             --no-check-certificate \
             "${url}&confirm=${confirm}" \
             -O "${dest}"
        rm -f "${cookie_file}"
    else
        # Regular HTTP download
        wget --no-check-certificate --show-progress -O "${dest}" "${url}"
    fi
}

verify_checksum() {
    local path=$1
    local dest="${MODEL_DIR}/${path}"
    
    echo -n "Verifying ${path}... "
    
    if [ ! -f "${dest}" ]; then
        echo "❌ File missing"
        return 1
    fi
    
    local checksum=$(sha256sum "${dest}" | awk '{print $1}')
    if [ "${checksum}" != "${MODEL_SHA256[$path]}" ]; then
        echo "❌ Checksum mismatch"
        echo "Expected: ${MODEL_SHA256[$path]}"
        echo "Got:      ${checksum}"
        return 1
    fi
    
    echo "✅"
    return 0
}

# Main download loop
failed_downloads=0
for model_path in "${!MODEL_URLS[@]}"; do
    dest_file="${MODEL_DIR}/${model_path}"
    
    if [ -f "${dest_file}" ]; then
        if verify_checksum "${model_path}"; then
            echo "Skipping ${model_path} (already exists and valid)"
            continue
        else
            echo "Redownloading ${model_path} (invalid checksum)"
            rm -f "${dest_file}"
        fi
    fi
    
    mkdir -p "$(dirname "${dest_file}")"
    
    if download_file "${MODEL_URLS[$model_path]}" "${dest_file}"; then
        if ! verify_checksum "${model_path}"; then
            rm -f "${dest_file}"
            failed_downloads=$((failed_downloads+1))
            echo "❌ Failed to verify ${model_path}"
        fi
    else
        failed_downloads=$((failed_downloads+1))
        echo "❌ Failed to download ${model_path}"
    fi
done

# Summary
if [ $failed_downloads -eq 0 ]; then
    echo "✅ All models downloaded and verified successfully"
else
    echo "❌ ${failed_downloads} models failed to download/verify"
    exit 1
fi

# Set permissions
chmod -R 755 "${MODEL_DIR}"
echo "Models are ready in ${MODEL_DIR}/"