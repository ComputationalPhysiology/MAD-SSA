#!/bin/bash

# Define parent and source folders
PARENT_FOLDER="/home/shared/00_data/"
SOURCE_FOLDER="/home/shared/Segmentations_ES/"

# Get folder name as input
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

FOLDER_NAME="$1"

# Extract the number after the underscore
NUMBER=$(echo "$FOLDER_NAME" | grep -oP '(?<=_)\d+')

if [ -z "$NUMBER" ]; then
    echo "Invalid folder name format. Expected format: <prefix>_<number>"
    exit 1
fi

# Create the new folder within the parent folder
NEW_FOLDER="$PARENT_FOLDER/$FOLDER_NAME"
mkdir -p "$NEW_FOLDER"
echo "Created folder: $NEW_FOLDER"

# Find the file in the source folder that starts with the extracted number
SOURCE_FILE=$(find "$SOURCE_FOLDER" -type f -name "${NUMBER}_*" | head -n 1)


if [ -z "$SOURCE_FILE" ]; then
    echo "No file found in $SOURCE_FOLDER starting with $NUMBER."
    exit 1
fi

# Copy the file to the new folder
cp "$SOURCE_FILE" "$NEW_FOLDER"
echo "Copied file: $SOURCE_FILE to $NEW_FOLDER"
