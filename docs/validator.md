- [Validator](#validator)
- [Setup and Usage](#setup-and-usage)
  - [Prerequisites](#prerequisites)
    - [Python environment with required dependencies](#python-environment-with-required-dependencies)
    - [Bittensor wallet](#bittensor-wallet)
  - [Running a Validator](#running-a-validator)

# 

**Note: This document applies to Bittensor Finney.**

If you are looking for guidance on local testing, please refer to the [local run](./local_test.md) documentation.



 

# Validator

Operating a validator node requires dedicated hardware and software resources. Validators play a critical role in the **SN SubQuery Hermes** network by:

- Generating synthetic challenges
- Evaluating and scoring miner performance
- Enhancing overall network security and reliability

Validator performance directly affects rewards: well-performing validators earn higher returns, while underperforming ones see their rewards reduced.

# Setup and Usage

## Prerequisites

- Python environment with required dependencies
- Bittensor wallet (coldkey and hotkey)
- Public IP for running validator

### Python environment with required dependencies

1、It is recommended to use `uv` with `python 3.13`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.13
```

2、clone `SN SubQuery Hermes`

```bash
git clone git@github.com:subquery/network-hermes-subnet.git

cd network-hermes-subnet

# sync and create venv
uv sync

source .venv/bin/activate

# install btcli
(network-hermes-subnet) uv pip install bittensor-cli 
```

### Bittensor wallet

We use `btcli` to create wallet.

1、Create a wallet

```bash
# this will need you to input your own password to proceed
(network-hermes-subnet) % 
btcli wallet new_coldkey --wallet.name validator
```

**Note:** This will generate a `coldkey` file in `~/.bittensor/wallets/validator`. Losing or exposing this file may compromise your funds. Keep it secure and private.

2、Create a hotkey

```bash
(network-hermes-subnet) % 
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

3、Register in `SN SubQuery Hermes`

```bash
(network-hermes-subnet) % 
btcli subnet register --wallet.name validator --wallet.hotkey default
```

If the registration is successful, you will receive a **UID**, which represents your hotkey slot in `SN SubQuery Hermes`.

**Note:** This operation requires a burn fee. Make sure your cold wallet has a sufficient TAO balance.

4、Become a Valid Validator

A validator’s stake is a crucial metric for Extraction in Bittensor. To qualify as a valid validator, your wallet must hold a sufficient stake.

<del> In `SN SubQuery Hermes`, at least **200 TAO** must be staked to register as a valid validator. </del>

As an option, you may perform a **self-stake**:

```bash
(network-hermes-subnet) % 
btcli stake add \
  --wallet.name validator \
  --wallet.hotkey default
```

## Running a Validator

Once everything is prepared, it’s time to launch the validator.

First, create a configuration file.

```bash
(network-hermes-subnet) %
cp .env.validator.example .env.validator
```

Second, edit the file to apply your own settings:

```ini
SUBTENSOR_NETWORK=finney
WALLET_NAME=validator
HOTKEY=default

# SN SubQuery Hermes NETUID
NETUID=10

# Your public IP address
EXTERNAL_IP=1.37.27.39
PORT=8085

# Board service base URL
BOARD_SERVICE=http://192.168.156.91:3000

OPENAI_API_KEY=sk-xxx

# For GraphQL agent & synthetic challenges
LLM_MODEL=gpt-5

# For scoring miners
SCORE_LLM_MODEL=o3
```

Configuration Parameters:

* `WALLET_NAME`: The identifier of your previously created cold wallet.

* `HOTKEY`: The identifier of your previously created hotkey wallet.

* `EXTERNAL_IP`: Your public IP address,  it serves as the entry point for other neurons to communicate with.

* `PORT`: Port corresponding to your `EXTERNAL_IP`.

* `BOARD_SERVICE`: URL from which the validator pulls projects.

* `OPENAI_API_KEY`: API key for OpenAI (currently the only supported provider).

* `LLM_MODEL`: LLM model used by the validator to generate synthetic challenges. GPT-5 or similar models are recommended.

* `SCORE_LLM_MODEL`: LLM model used by the validator to score miners. It is recommended to use a model with reasoning capabilities, such as `o3`.

<br />

Last,  launch the Validator：

```bash
(network-hermes-subnet) % 
python -m neurons.validator
```

This will pull projects and start serving. You should see output similar to the following:

```bash
2025-09-04 11:56:58.331 | INFO     | __main__:serve_api:117 - Starting serve API on http://0.0.0.0:8085
INFO:     Started server process [73390]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8085 (Press CTRL+C to quit)
```
