# 1、Run subtensor locally

```shell
git clone https://github.com/opentensor/subtensor.git

cd subtensor

vi pallets/subtensor/src/subnets/registration.rs
# modify the difficulty to 1 in do_faucet function
#################
pub fn do_faucet(
        origin: T::RuntimeOrigin,
        block_number: u64,
        nonce: u64,
        work: Vec<u8>,
    ) -> DispatchResult {
        ...
        // --- 3. Ensure the supplied work passes the difficulty.
        // let difficulty: U256 = U256::from(1_000_000); // Base faucet difficulty.
        let difficulty: U256 = U256::from(1); // Base faucet difficulty.
       ...
}
#################

./subtensor/scripts/init.sh


BUILD_BINARY=1 ./scripts/localnet.sh False
```

## 2、Prepare env & Install btcli

```shell
mkdir env_test

cd env_test

uv venv --python 3.12

source .venv/bin/activate

(env_test)[env_test] git clone https://github.com/opentensor/btcli.git

(env_test)[env_test] cd btcli

(env_test) [btcli] uv pip install -e . 
```

# 3、Prepare wallet

## owner

 coldkey：

```shell
(env_test) [btcli] % 
btcli wallet new_coldkey --wallet.name owner

Enter the path to the wallets directory (~/.bittensor/wallets/): 
Choose the number of words [12/15/18/21/24]: 

IMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone who has possession of this mnemonic can use it to regenerate the key and access your tokens.

The mnemonic to the new coldkey is: fit aunt smooth wood approve stone flat swear knock dove version master
You can use the mnemonic to recreate the key with `btcli` in case it gets lost.

input password（123456）
```

hotkey：

```shell
(env_test) [btcli] % 
btcli wallet new_hotkey --wallet.name owner --wallet.hotkey default
```

will generate structure like：

```html
~/.bittensor/wallets
├── owner
│   ├── coldkey
│   ├── coldkeypub.txt
│   └── hotkeys
│       ├── default
│       └── defaultpub.txt
```

## Validator

```shell
(env_test) [btcli] % 
btcli wallet new_coldkey --wallet.name validator


(env_test) [btcli] % 
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

## Miner

```shell
(env_test) [btcli] % 
btcli wallet new_coldkey --wallet.name miner


(env_test) [btcli] % 
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

# 4、mint token from faucet

owner：

```shell
(env_test) [btcli] % 
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9944

Run Faucet?
 wallet name: owner
 coldkey:    5DnfkGKdfAXfiqJ8xsC3sgV9N52hNQk2zcAbXhq35A8YFYvV
 network:    Network: local, Chain: ws://127.0.0.1:9944 [y/n]: y
Enter your password: 
Decrypting...
Balance: ‎0.0000 τ‎ ➡ ‎1,000.0000 τ‎

Balance: ‎1,000.0000 τ‎ ➡ ‎2,000.0000 τ‎


^C
```

validator：

```shell
(env_test) [btcli] % 
btcli wallet faucet --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9944
```

miner：

```shell
(env_test) [btcli] % 
btcli wallet faucet --wallet.name miner --subtensor.chain_endpoint ws://127.0.0.1:9944
```

# 5、Create subnet & Register Validator/Miner

create subnet with owner（get netuid 2）：

```shell
(env_test) [btcli] % 
btcli subnet create --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9944

Enter the wallet hotkey (Hint: You can set this with `btcli config set --wallet-hotkey`) (default): 
Subnet name (optional) (): first_sub
GitHub repository URL (optional) (): 
Contact email (optional) (): 
Subnet URL (optional) (): 
Discord handle (optional) (): 
Description (optional) (): 
Additional information (optional) (): 
Subnet burn cost: ‎1,000.0000 τ‎
Your balance is: ‎2,000.0000 τ‎
Do you want to burn ‎1,000.0000 τ‎ to register a subnet? [y/n]: y
Enter your password: 
Decrypting...
✅ Registered subnetwork with netuid: 2
Would you like to set your own identity? [y/n]: y
Existing identity not found for 5DnfkGKdfAXfiqJ8xsC3sgV9N52hNQk2zcAbXhq35A8YFYvV on Network: local, Chain: ws://127.0.0.1:9944

Cost to register an Identity is 0.1 TAO, are you sure you wish to continue? [y/n]: n
❌ Aborted!
```

register miner（miner get uid 1）：

```shell
(env_test) [btcli] % 
btcli subnet register --wallet.name miner --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9944


            Register to netuid: 2                                                   
            Network: local                                                       

 Netu… ┃ Sym… ┃ Cost (… ┃                     Hotkey                     ┃                     Coldkey                     
━━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   2   │  β   │ τ 0.55… │ 5FxX4PS6Jfis7HJCBqkUjSQmYh94FJy44B7eUxGVj8oz5… │ 5DS7X1Yx2cJ32zXnDXxat8yu33cadW1BZYCAFdUXCs5mFd… 
───────┼──────┼─────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────
       │      │         │                                                │                                                 
Your balance is: ‎3,000.0000 τ‎
The cost to register by recycle is ‎0.5546 τ‎
Do you want to continue? [y/n] (n): y
Enter your password: 
Decrypting...
Balance:
  ‎3,000.0000 τ‎ ➡ ‎2,999.4454 τ‎
✅ Registered on netuid 2 with UID 1
```

register validator（validator get uid 2）：

```powerquery
(env_test) [btcli] % 
btcli subnet register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9944


        Register to netuid: 2                                                       
        Network: local                                                          

 Netuid ┃ Symbol ┃ Cost (Τ) ┃                      Hotkey                      ┃                     Coldkey                      
━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   2    │   β    │ τ 0.5546 │ 5HKLAobNfkSkvjaEuePNPm6f3KNAFXsqUE3oRaPR5jFMuigC │ 5EjrfNRDunhptvvjRxeWpapT9FDpwymK4h3iAZRrTQsN8VbW 
────────┼────────┼──────────┼──────────────────────────────────────────────────┼──────────────────────────────────────────────────
        │        │          │                                                  │                                                  
Your balance is: ‎3,000.0000 τ‎
The cost to register by recycle is ‎0.5546 τ‎
Do you want to continue? [y/n] (n): y
Enter your password: 
Decrypting...
Balance:
  ‎3,000.0000 τ‎ ➡ ‎2,999.4454 τ‎
✅ Registered on netuid 2 with UID 2 
```

# 6、Run hermes

1、clone

```shell
(env_test) [btcli] % cd ../

(env_test) [env_test] git@github.com:subquery/network-hermes-subnet.git

(env_test) [env_test] cd network-hermes-subnet

(env_test) [env_test] git checkout feat/develop

(env_test) [network-hermes-subnet] uv sync
```

2、modify path：

 modify `agent/graphql_agent.py`：

```python
import os
import sys

# change to your local subql-graphql-agent path
sys.path.append(os.path.abspath("/Users/demon/Desktop/work/onf/subql-graphql-agent/examples"))

...
```

3、`subql-graphql-agent`add a function.

navigate to`subql-graphql-agent/examples/server.py`

```python
class GraphQLAgent:
    """GraphQL agent for a specific SubQuery project."""

    ...

    async def query_no_stream(self, question):
        response = await self.executor.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={
                "recursion_limit": 25,
            }
        )
        return response
```

4、run validator：

* add `.env.validator` file

```ini
SUBTENSOR_NETWORK=ws://127.0.0.1:9944
WALLET_NAME=validator
HOTKEY=default
NETUID=2

# change to en0 address
EXTERNAL_IP=192.168.1.60
PORT=8085

# change
BOARD_SERVICE=http://192.168.156.91:3000
OPENAI_API_KEY=sk-xx


# for graphql agent & synthetic challenge
LLM_MODEL=gpt-5

# for score
SCORE_LLM_MODEL=o3
```

* run：

```shell
(env_test) [network-hermes-subnet] python -m neurons.validator
```

5、run miner：

* add `.env.miner` file

```ini
SUBTENSOR_NETWORK=ws://127.0.0.1:9944
WALLET_NAME=miner
HOTKEY=default
NETUID=2

# change to en0 address
EXTERNAL_IP=192.168.1.60
PORT=8086

# change
BOARD_SERVICE=http://192.168.156.91:3000
OPENAI_API_KEY=sk-xx


# miner self-owned agent
MINER_LLM_MODEL=gpt-4o-mini

# for fallback graphql agent
LLM_MODEL=gpt-4o
```

* run：

```shell
(env_test) [network-hermes-subnet] python -m neurons.miner
```
