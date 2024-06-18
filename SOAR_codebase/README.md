# auto-improv-orchestrator

## Installation
1.
    ```
    conda create -n orchestrator python=3.10
    pip install -r requirements.txt
    pip install -e .
    ```
2. You'll also have to manually install Jax for your platform (see the [Jax installation instructions](https://jax.readthedocs.io/en/latest/installation.html)).
3. Save your OpenAI API key to the environment variable OPENAI_API_KEY
4. Install the google cloud CLI

## Running Instructions

### Robot autonomous data collection
```
python orchestrator/robot/main.py --config_dir config/<path to config dir>
```
