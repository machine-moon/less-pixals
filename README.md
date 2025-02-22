# JAX on Cloud TPU VM

This project demonstrates how to run JAX code on a Cloud TPU VM. It includes examples and scripts to set up the environment, synchronize code, and execute training tasks on the TPU.

## Requirements

- Google Cloud account
- Google Cloud SDK installed
- Cloud TPU VM setup
- Python 3.7+
- JAX and other dependencies (see `requirements.txt`)

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-repo/jax-tpu-vm.git
    cd jax-tpu-vm
    ```

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    Create a `secrets` file with the following content:

    ```sh
    export ACCELERATOR_TYPE=v4-16
    export RUNTIME_VERSION=tpu-ubuntu2204-base
    export ZONE=us-central2-b
    export TPU_NAME=node-1
    export PROJECT_ID=your-project-id
    export RUNTIME_SECRETS="secrets"
    export SRC_DIR="./"
    export DEST_DIR="./jax-tpu-vm"
    export EXECUTABLE="main.py"
    ```

    Source the environment variables:

    ```sh
    source ./secrets
    ```

## Running the Code

1. **Sync code to TPU VM:**

    ```sh
    make sync
    ```

2. **Run the training script:**

    ```sh
    make run
    ```

3. **Monitor the training process:**

    ```sh
    make ssh0
    tail -f /tmp/jax-tpu-vm/output.log
    ```

## Example Output

The training script will output logs similar to the following:

```
INFO:absl:train epoch: 1, loss: 0.2352, accuracy: 92.94
INFO:absl:eval epoch: 1, loss: 0.0592, accuracy: 98.00
INFO:absl:train epoch: 2, loss: 0.0584, accuracy: 98.15
INFO:absl:eval epoch: 2, loss: 0.0575, accuracy: 98.14
INFO:absl:train epoch: 3, loss: 0.0423, accuracy: 98.66
INFO:absl:eval epoch: 3, loss: 0.0357, accuracy: 98.78
```

## Cleaning Up

To clean up the TPU VM and remove synced files:

```sh
make kill
make __delete
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.