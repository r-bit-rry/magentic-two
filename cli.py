import argparse
import asyncio
import os
from autogen_agentchat.ui import Console

from autogen_ext.teams.magentic_one import MagenticOne
from ollama_chat_client import create_local_completion_client


def main() -> None:
    """
    Command-line interface for running a complex task using MagenticOne.

    This script accepts a single task string and an optional flag to disable
    human-in-the-loop mode. It initializes the necessary clients and runs the
    task using the MagenticOne class.

    Arguments:
    task (str): The task to be executed by MagenticOne.
    --hil: Optional flag to disable human-in-the-loop mode.

    Example usage:
    python magentic_one_cli.py "example task"
    python magentic_one_cli.py -hil "example task"
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a complex task using MagenticOne.\n\n"
            "For more information, refer to the following paper: https://arxiv.org/abs/2411.04468"
        )
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="?",
        help="The task to be executed by MagenticOne.",
        default="I want you to pick 5 stocks that you think will perform well in the next 6 weeks. Please provide a brief explanation for each stock. I want you to go over reddit.com/r/wallstreetbets and the website stockwits along with twitter to find the most discussed stocks. The stocks need to be cheap still, that is, close to the value at IPO, have not been pump and dump yet, and the companies are not bankrupt or shady. you need to pick stocks for holding positions only, no shorts, no options. Good examples for stocks that worked well recently are: RKLB, ASTS, LUNR, MTEK, SATL. Only write and use code if you find it necessary, most of the tasks can be done by reading and summarizing information.",
    )
    parser.add_argument(
        "--hil", action="store_false", help="Enable human-in-the-loop mode."
    )
    args = parser.parse_args()
    async def run_task(task: str, hil_mode: bool) -> None:
        client = create_local_completion_client()
        m1 = MagenticOne(client=client, hil_mode=hil_mode)
        await Console(m1.run_stream(task=task))

    task = args.task[0]
    task =  """I want you to pick 5 stocks that you think will perform well in the next 6 weeks.
    I want you to go over stocktwits.com and other relevant market news webstites.
    Find the most discussed stocks.
    The stocks need to be cheap still, that is, close to the value at IPO, have not been pump and dump yet, and the companies are not bankrupt or shady.
    You need to pick stocks for holding positions only, no shorts, no options.
    Good examples for stocks that worked well recently are: RKLB, ASTS, LUNR, MTEK, SATL. Only write and use code if you find it necessary, most of the tasks can be done by reading and summarizing information.
    Provide timestamps for retrieved data, provide reasoning for choosing the stocks, provide latest economic news about those stocks.
    The only API_KEY available for us is for BING search engine as BING_API_KEY={bing_api_key}. Use it inside of your code to fetch relative information
    POrovide a list of the stocks by ticker, current price, timestamp of retrieval, quick economic review.
    """.format(bing_api_key=os.environ.get("BING_API_KEY"))
    asyncio.run(run_task(task, not args.hil))


if __name__ == "__main__":
    main()
