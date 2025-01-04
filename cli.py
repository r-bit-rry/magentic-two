import argparse
import asyncio
from autogen_agentchat.ui import Console

from magenticone import MagenticOne
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
        default="test",
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
    task = """Be opinionated, technical, concise, adverserial and risk taking.
    I want you to pick 10 stocks that you think will perform well in the next 2 weeks and match a strategy of buying and holding for a few days or weeks.
    Only buy and hold positions, no shorts, no options.
    I want you to go over twitter/x mentions of ticker names and other relevant market news websites.
    Find the newly discussed stocks, newly followed or watched over stocks that get the market attention, try to crawl and see as many relevant stocks as you can.
    Try to find economic trends and new market trends, for example quantum, AI, chip manufacturing, nuclear power and so on.
    The stocks should be small-cap up to 3 billion dollar in value, stock price close to or lower than the value at IPO with no splits, have not been pump and dump yet, no holding companies, companies with real product, contracts, employess.
    Examples of stocks matching that: RKLB, ASTS, LUNR, MTEK, SATL, NUKK, KULR, NITO, LPTH, QBTS, QNCCF.
    Only write and use python code if you find it necessary, most of the tasks can be done by reading and summarizing information.
    Extract datetime from the content you used while searching, provide reasoning for choosing the stocks, provide latest economic news about those stocks.
    Use Bing Search tool to get the latest news, its tuned to search only for the last week.
    Use the URL returned by the Search tool and direct the websurfer there.
    Provide a list of the stocks by ticker, current price, date and time of evaluation, quick economic review.
    """
    asyncio.run(run_task(task, not args.hil))


if __name__ == "__main__":
    main()
