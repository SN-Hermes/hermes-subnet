# Running Hermes Subnet on Testnet

Please first follow the steps in [Running Subnet Locally](./local_test.md) to prepare your environment and wallets.

If you have already prepared everything, running on testnet becomes much easier.

## Hermes Netuid on Testnet

Hermes runs on netuid `280` in testnet.

You can check it using the btcli command:

```shell
btcli subnet show --netuid 280 --network test
```

## Getting Tao

Assuming you have completed the local run and have wallets set up. You will need some testnet tao. Fortunately, Bittensor provides an official faucet to help us get test tao.

Faucet URL: https://app.minersunion.ai/testnet-faucet

Enter your wallet address and wait a moment to receive tao.

## Registration

With tao, you can register as a Validator or Miner on testnet.
Use the same commands as local run, but change the endpoint to the testnet endpoint:

```shell
btcli subnet register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint wss://test.finney.opentensor.ai:443
```

Note: Bittensor network addresses can be found in their official documentation: https://docs.learnbittensor.org/concepts/bittensor-networks

## Running

Same as local run, you just need to change the endpoint address to testnet and netuid:

### .env.validator

```ini
SUBTENSOR_NETWORK=wss://test.finney.opentensor.ai:443
NETUID=280
```

Keep other configurations unchanged, then run:

```shell
python -m neurons.validator
```

### .env.miner

```ini
SUBTENSOR_NETWORK=wss://test.finney.opentensor.ai:443
NETUID=280
```

Keep other configurations unchanged, then run:

```shell
python -m neurons.miner
```
