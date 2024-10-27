import asyncio




completer_name = (  # Weak Completer
    "command-r-03-2024"  # Instruction-following conversational model (128k ctx)
)
strong_verifier_name = (
    "command-r-plus-08-2024"  # Most capable as of 10/12/2024 (128k ctx)
)


async def main():
    # Read the text file of row_ids that are solveable

    # Read these rows from cn_k12 using the utils.read_specific_rows function

    # For each, generate N prefixes using the weak completer (will lolook ery similar to remaining_on_policy, but no concatenation with existing problems)


    ...


if __name__ == "__main__":
    asyncio.run(main())