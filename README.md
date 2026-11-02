# DockerXGBoostGPU

Private research repository for experiments on using XGBoost with GPU acceleration.

## Objectives

- Run reproducible XGBoost experiments using NVIDIA GPUs.
- Support execution through Docker in local and remote GPU environments.

## Project structure

- `src/`: Python source code.
- `datasets/`: local training and test datasets; not tracked by Git.
- `results/`: generated results and logs; not tracked by Git.
- `Dockerfile`: Docker image configuration.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.14.4
- Docker with NVIDIA GPU support.
- Compatible NVIDIA GPU drivers and container runtime.

## Data

Place the training and test CSV files in the local `datasets/` directory. These files are excluded from version control by default.

## Reproducibility

Record the random seeds, model parameters, software versions and dataset versions used in each experiment.


## Docker Usage

### 1. Build the Docker Image

Build the Docker image with both a version-specific tag and the `latest` tag.

```bash
docker build -t xgb-xai-stability:1.0 -t xgb-xai-stability:latest .
```

### 2. Run the Container

Run the experiment using the `optuna` strategy and the specified configuration and datasets.

```bash
docker run -d \
  --name xgb_fifa \
  --gpus device=4 \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/config:/app/config:ro" \
  -v "$PWD/results:/app/results" \
  xgb-xai-stability:1.0 \
    --strategy optuna \
    --runs 100 \
    --seed 42 \
    --config config/config.json \
    --train data/fifa/train_fifa.csv \
    --test data/fifa/test_fifa.csv \
    --sep comma \
    --output-dir results/fifa
```

#### Command Options

| Option | Description |
|---|---|
| `--name` | Assigns a name to the container. |
| `--gpus device=4` | Makes GPU 4 available to the container. |
| `--user` | Runs the process using the host user's UID and GID. |
| `-v ...:ro` | Mounts input data and configuration directories as read-only. |
| `--strategy optuna` | Selects the hyperparameter optimization strategy. |
| `--runs 100` | Sets the number of experiment runs. |
| `--seed 42` | Sets the random seed. |
| `--config` | Specifies the JSON configuration file. |
| `--train` / `--test` | Specify the training and test datasets. |
| `--sep comma` | Sets the CSV field separator to a comma. |
| `--output-dir` | Specifies the output directory. |

**Note:** Run these commands from the project root directory. Ensure that GPU 4 exists and is available on the host. The `data` and `config` directories must exist before starting the container.

### 3. Monitor the Logs

Follow the container logs in real time.

```bash
docker logs -f xgb_fifa
```

Press `Ctrl+C` to stop following the logs. This does not stop the container.

### 4. Stop the Container

Stop the container if it is still running.

```bash
docker stop xgb_fifa
```

### 5. Remove the Container

Remove the container after the experiment has finished.

```bash
docker rm xgb_fifa
```
 


## License

Not specified yet.