import asyncio
from loguru import logger

from common.protocol import SyntheticNonStreamSynapse
from neurons.miner import Miner

async def main():
    miner = Miner() 
    local_agents = await miner.refresh_agents(pull=False)

    while True:
        try:
            local_projects_cids = list(local_agents.keys())
            print("\nAvailable projects:")
            for idx, cid in enumerate(local_projects_cids, start=1):
                print(f"{idx}) {cid}")

            selected_index = input("\nplease select a project: ").strip()
            if not selected_index.isdigit() or int(selected_index) < 1 or int(selected_index) > len(local_projects_cids):
                print("❌ 无效的选择，请重新输入！")
                continue

            selected_cid = local_projects_cids[int(selected_index) - 1]
            print(f"\n✅ you selected: {selected_cid}")

            question = input("\n🙋 input replay challenge: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            synapse = SyntheticNonStreamSynapse(id='relay-01', project_id=selected_cid, question=question)
            # await miner.forward_synthetic_non_stream(synapse)
            response = await miner.invoke_server_agent(synapse)
            print(f"\n🤖 invoke_server_agent response: {response}\n")

            response = await miner.invoke_miner_agent(synapse)
            print(f"\n🤖 invoke_miner_agent response: {response}\n")

            #TODO: score

        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())