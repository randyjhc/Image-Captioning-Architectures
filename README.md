# Image-Captioning-Architectures

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Fast Python package installer and resolver
- pre-commit - Git hook scripts for identifying simple issues before submission

## Usage

### Installing Development Packages

Install all development dependencies using uv:

```bash
uv sync --dev
```

This will install all dependencies including development tools like mypy and pre-commit.

### Setting Up Pre-commit Hooks

After installing the development packages, set up pre-commit hooks:

```bash
pre-commit install
```

This will configure the following hooks to run automatically before each commit:
- **ruff** - Fast Python linter and formatter (with auto-fix)
- **mypy** - Static type checker

### Running Pre-commit Manually

To manually run pre-commit on all files before committing:

```bash
pre-commit run --all-files
```

Or to run on staged files only:

```bash
pre-commit run
```

## Dataset Setup

### Downloading the Flickr8k Dataset

This project uses the Flickr8k dataset, which contains 8,000 images with 5 captions each.

#### 1. Set Up Kaggle API Credentials

Before downloading the dataset, you need to configure your Kaggle API credentials:

1. Go to [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Scroll to the "API" section and click "Create New API Token"
3. This will automatically download a `kaggle.json` file to your Downloads folder
4. Move the downloaded file to the correct location:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

#### 2. Download the Dataset

Run the download script:

```bash
./scripts/download_flickr8k.sh
```

The dataset will be downloaded and extracted to `data/datasets/flickr8k/`.

**Note:** The dataset is approximately 1GB in size and may take several minutes to download depending on your internet connection.