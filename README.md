<div align="center">

# **SubQuery Hermes Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Decentralized GraphQL Query Infrastructure <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [SubQuery Network](https://subquery.network/) • [Documentation](./docs/)
</div>

---
- [Introduction](#introduction)
  - [What is SubQuery Hermes Subnet?](#what-is-subquery-hermes-subnet)
  - [How It Works](#how-it-works)
- [Installation](#installation)
  - [Before you proceed](#before-you-proceed)
  - [Install](#install)
- [For Validators](#for-validators)
  - [Validator Overview](#validator-overview)
  - [Running a Validator](#running-a-validator)
- [For Miners](#for-miners)
  - [Miner Overview](#miner-overview)
  - [Running a Miner](#running-a-miner)
  - [Optimizing Performance](#optimizing-performance)
- [Documentation](#documentation)
- [License](#license)

---

## Introduction

**IMPORTANT**: If you are new to Bittensor subnets, read this section before proceeding to [Installation](#installation) section.

### What is SubQuery Hermes Subnet?

SubQuery Hermes Subnet is a specialized Bittensor subnet that creates a decentralized infrastructure for GraphQL query processing and blockchain data indexing. It leverages the power of the Bittensor network to incentivize high-performance, accurate GraphQL query responses across multiple blockchain ecosystems.

The subnet focuses on:
- **GraphQL Query Processing**: Miners compete to provide fast, accurate responses to GraphQL queries
- **Blockchain Data Indexing**: Support for SubQuery projects and The Graph subgraphs
- **Multi-Chain Support**: Query data across various blockchain networks
- **Performance Optimization**: Incentivizing response speed while maintaining data accuracy

### How It Works

The SubQuery Hermes Subnet operates through a sophisticated incentive mechanism:

1. **Project Selection**: Validators select active projects from the [SubQuery Network board](https://hermes.subquery.network/)
2. **Synthetic Challenges**: Validators generate GraphQL-based questions using project schemas
3. **Miner Competition**: Miners receive queries and compete to provide fast, accurate responses
4. **Performance Evaluation**: Responses are evaluated based on:
   - **Factual Accuracy**: Correctness compared to ground truth
   - **Response Time**: Speed of query processing
   - **Query Optimization**: Efficiency of GraphQL operations

5. **Reward Distribution**: TAO rewards are distributed based on performance metrics, encouraging continuous optimization

**Key Components:**
- **Subnet Validators**: Generate synthetic challenges, evaluate responses, and maintain consensus
- **Subnet Miners**: Process GraphQL queries, optimize response times, and compete for rewards
- **GraphQL Agents**: AI-powered agents that understand project schemas and generate optimized queries
- **Project Management**: Dynamic loading and management of SubQuery and The Graph projects

---

## Installation

### Before you proceed
Before you proceed with the installation of the subnet, note the following: 

- Use these instructions to run your subnet locally for your development and testing, or on Bittensor testnet or on Bittensor mainnet. 
- **IMPORTANT**: We **strongly recommend** that you first run your subnet locally and complete your development and testing before running the subnet on Bittensor testnet. Furthermore, make sure that you next run your subnet on Bittensor testnet before running it on the Bittensor mainnet.
- You can run your subnet either as a subnet owner, or as a subnet validator or as a subnet miner. 
- **IMPORTANT:** Make sure you are aware of the minimum compute requirements for your subnet. See the [Minimum compute YAML configuration](./min_compute.yml).
- Note that installation instructions differ based on your situation: For example, installing for local development and testing will require a few additional steps compared to installing for testnet. Similarly, installation instructions differ for a subnet owner vs a validator or a miner. 

### Install

- **Running locally**: Follow the step-by-step instructions described in this section: [Running Subnet Locally](./docs/local_test.md).
- **Running on Bittensor testnet**: Follow the step-by-step instructions described in this section: [Running on the Test Network](./docs/running_on_testnet.md).
- **Running on Bittensor mainnet**: Follow the step-by-step instructions described in this section: [Running on the Main Network](./docs/running_on_mainnet.md).


## For Validators

### Validator Overview

Validators in SubQuery Hermes Subnet are responsible for:
- **Project Management**: Monitoring and selecting active SubQuery/The Graph projects
- **Challenge Generation**: Creating synthetic GraphQL queries based on project schemas  
- **Response Evaluation**: Scoring miner responses based on accuracy and performance
- **Consensus Participation**: Contributing to the network's consensus mechanism
- **Network Health**: Ensuring the overall quality and reliability of the subnet

### Running a Validator

Validators require significant computational resources and stake to participate effectively. They must:
- Maintain high uptime and reliability
- Generate sophisticated challenges that test miner capabilities
- Evaluate responses fairly and accurately
- Contribute to network consensus

**Quick Start:**
```bash
# Setup environment
uv sync
source .venv/bin/activate

# Configure validator settings
cp .env.validator.example .env.validator
# Edit .env.validator with your configuration

# Run validator
python -m neurons.validator
```

**LLM Configuration:**
- Use `OPENAI_BASE_URL` environment variable to configure alternative LLM providers (e.g., local models, other API endpoints)
- **Important for Validators**: Avoid using GPT mini series models (gpt-4o-mini, etc.) as they have been tested and shown to perform poorly for challenge generation and response evaluation
- Tested models (in openrouter): google/gemini-3-flash-preview, z-ai/glm-4.7

For detailed validator setup instructions, see [Validator Documentation](./docs/validator.md).

## For Miners

### Miner Overview

Miners in SubQuery Hermes Subnet compete by:
- **Query Processing**: Efficiently handling GraphQL queries across multiple projects
- **Response Optimization**: Minimizing response time while maintaining accuracy
- **Tool Development**: Creating specialized tools for specific project types
- **Continuous Learning**: Adapting to new projects and query patterns
- **Resource Management**: Balancing computational resources for optimal performance

### Running a Miner

Miners can participate with varying levels of optimization:
- **Basic Setup**: Run with default GraphQL agent capabilities
- **Optimized Setup**: Develop custom tools for specific projects
- **Advanced Setup**: Implement sophisticated query optimization strategies

**Quick Start:**
```bash
# Setup environment  
uv sync
source .venv/bin/activate

# Configure miner settings
cp .env.miner.example .env.miner
# Edit .env.miner with your configuration

# Run miner
python -m neurons.miner
```

### Optimizing Performance

To maximize rewards, miners should:
1. **Analyze Project Schemas**: Understand the data structures and relationships
2. **Develop Custom Tools**: Create project-specific query optimization tools
3. **Monitor Performance**: Track response times and accuracy metrics
4. **Iterate and Improve**: Continuously refine tools based on challenge patterns

For comprehensive miner optimization guides, see [Miner Documentation](./docs/miner.md).

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Local Testing Guide](./docs/local_test.md)**: Instructions for running the subnet locally for development and testing
- **[Validator Guide](./docs/validator.md)**: Detailed validator setup, configuration, and operation
- **[Miner Guide](./docs/miner.md)**: Complete miner setup, optimization strategies, and tool development
- **[Incentive Mechanism](./docs/incentive.md)**: Deep dive into the subnet's scoring and reward system

Additional resources:
- [SubQuery Network](https://subquery.network/): Learn more about SubQuery's decentralized data infrastructure
- [The Graph](https://thegraph.com/): Information about The Graph protocol and subgraphs
- [Bittensor Documentation](https://docs.bittensor.com/): General Bittensor network documentation

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```