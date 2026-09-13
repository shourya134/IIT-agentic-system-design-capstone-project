import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from crew.crew_setup import run_crew  # noqa: E402
from crew.output_schema import ResponseSchema, ValidationError, parse_crew_output  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 2, Task 9: structured output schema (ResponseSchema) validated in code")

    response = run_crew("What documents do I need for KYC verification?")
    log(f"Crew response validated against ResponseSchema: {response.model_dump()}")
    assert isinstance(response, ResponseSchema)
    log("PASS: crew output validates against ResponseSchema.")

    log("Deliberately malformed output (missing required fields):")
    malformed_json = '{"answer": "some text"}'
    try:
        parse_crew_output(malformed_json)
        log("ERROR: malformed output was NOT rejected")
    except ValidationError as e:
        log(f"PASS: malformed output correctly rejected by validation: {e.errors()[:1]}")

    log.save("part2_task9.txt")


if __name__ == "__main__":
    main()
