# triage-keepers

## Introduction

Triage Keepers is a program designed to analyze portrait photos. 

The program performs an assessment on the sharpness of photos, identifies similarities between photos and grouped in "bursts", maintains the metadata of the photo if the RAW and .jpg/.jpeg are found together, and provides a statistics summary for all of the folders available.

## Install Instructions

In the spirit of utilizing AI, follow the instructions provided with CLAUDE.md for local utilization.

- Install the uv package. Use the curl command below or any other package manager based on preference

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To use triage-keepers within Claude, a few steps are required:

1. Add the following code block to the tools config under the developer tab
    - "command" needs to be updated with the appropriate path to your uv module
    - under the "--project" section, update the /path/to/ points with the folder location of this git repo.

```json

{
  "mcpServers": {
    "triage-keepers": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project",
        "/path/to/triage-keepers",
        "python",
        "/path/to/triage-keepers/server.py"
      ]
    }
  }
}
```

2. Highly recommended to not automatically override the file, this will remove other MCPserver connections that might pre-exist


## Run Instructions

Running this once the tool is available under the tools section requires two items for setup outside of the code. The user needs to identify the appropriate path where the pictures reside.

Then call triage-keepers with the path.

## Available Methods

Below are the 12 methods that stack into the features listed above:

| Method Name     | Description |
| :---            |        :---:|
|list_folders | Lists all subfolders recursively through each defined list |
| get_thumbnail | Takes images from the folders and returns them. If user adds a subnote to annotate faces, the picture will be returned with new boxes drawn for face identification |
| index_folder| Loops through the folders and subfolders recursively in the given path |
| summarize_folder| Summarizes the stats of the current cached db |
| asses_subject_sharpness | Uses Laplacian algorithms from cv2 module to determine sharpness of individual pictures |
| find_unsharp_subjects | Finds the lowest percentile if relative. Finds all objects with sharpness less than 50% - which can be adjusted |
| find_no_subjects | Pulls photos that didn't have a face or eyes to detect |
| get_metadata | Pulls metadata of a cached photo through its path. |
| find_burst_groups | Identifies groups of photos that are near duplicates |
| rank_burst_group | Ranks burst groups - WIP - further adjustment required on photo collection |
| get_pair | JPG/JPEG can be paired with a RAW file - Canon RAW files have .CR2 extension |
| find_orphans | Opposite of get_pair, finds RAW or standard picture doesn't have a matching opposing file and returns |




## Current Status

Rank burst groups requires adjustments. Project requires manual scoring in the future

