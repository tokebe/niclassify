import sys
from pathlib import Path

import httpx
from pydantic import BaseModel

from niclassify.config.general import CONFIG
from niclassify.core.interfaces.handler import Handler

CLIENT = httpx.Client(
    transport=httpx.HTTPTransport(retries=3), timeout=120, follow_redirects=True
)

# TODO: The below will be useful eventually when bold v4 is cut off.
# Currently, it's not worth pursuing as bold v5 seems incredibly slow.
# See https://boldsystems.org/data/api/#section1
# UPDATE 2026-02-27: BOLD v5 is now fast :)


class PreProcessorResponseTerm(BaseModel):
    """A preprocessor search term result."""

    submitted: str
    matched: str


class PreProcessorResponse(BaseModel):
    """Response format to /api/query/preprocessor.

    See https://portal.boldsystems.org/api/docs#/query/resolve_query_api_query_preprocessor_get
    """

    successful_terms: list[PreProcessorResponseTerm] | None = None
    failed_terms: list[PreProcessorResponseTerm] | None = None


def select_terms(body: PreProcessorResponse, handler: Handler) -> str:
    """Select desired terms from the match response."""
    final_terms = list[str]()

    if body.successful_terms is None:
        raise TypeError(
            "BOLD Preprocess query succeeded, but no successful terms were returned."
        )

    for term in body.successful_terms:
        prefix = term.submitted.split(":")[0]
        options = list[str]()
        for match in term.matched.split(";"):
            if prefix in match:
                options.append(match)
        if len(options) == 0:
            handler.error(
                f"No appropriate matches were found for term {term.submitted} (Matched terms included: {term.matched.split(';')}).\nTry again with different search terms, bearing in mind that BOLD is case-sensitive.",
                abort=True,
            )
        elif len(options) == 1:
            final_terms.append(options[0])
        else:
            selection = handler.select(
                f"Select preferred match for search term `{term.submitted}`",
                options,
                abort=True,
            )
            final_terms.append(selection)
    handler.debug(str(final_terms))
    return ";".join(final_terms)


def get_search_terms(
    geography: str, taxonomy: str, handler: Handler
) -> PreProcessorResponse:
    """Refine user searchterms to a BOLD-approved query string."""
    if geography.islower():
        handler.log(
            f"Search geography `{geography}` is lower-case. BOLD is case-sensitive and usually prefers titlecase."
        )
        if handler.confirm(
            f"Proceed with titlecased search term (`{geography.title()}`?)"
        ):
            geography = geography.title()
    if taxonomy.islower():
        handler.log(
            f"Search taxonomy `{taxonomy}` is lower-case. BOLD is case-sensitive and usually prefers titlecase."
        )
        if handler.confirm(
            f"Proceed with titlecased search term (`{taxonomy.title()}`?)"
        ):
            taxonomy = taxonomy.title()

    with handler.spin() as spinner:
        handler.debug(f"Using BOLD host {CONFIG.apis.bold.host}")
        task = spinner.add_task(
            description="Checking search terms with BOLD...", total=1
        )
        try:
            url = f"{CONFIG.apis.bold.host}/query/preprocessor?query=geo:{geography};tax:{taxonomy}"
            response = CLIENT.get(url).raise_for_status()
            handler.debug(str(response.json()))
            body = PreProcessorResponse.model_validate(response.json())
            spinner.update(
                task, description="Checking search terms with BOLD...done.", completed=1
            )
            return body

        except httpx.HTTPStatusError as error:
            body = PreProcessorResponse.model_validate(error.response.json())
            if error.response.status_code != 400:  # noqa:PLR2004 it's HTTP 400
                handler.error(str(error))
                handler.error(
                    f"BOLD Preprocess query returned with failing HTTP status {error.response.status_code}. See above for error details.",
                    abort=True,
                )
            if body.failed_terms is None:
                handler.error(str(error))
                handler.error(str(error.response))
                handler.error(
                    "BOLD Preprocess query failed to process for unknown reasons, see above for error details.",
                    abort=True,
                )
                sys.exit(1)

            for term in body.failed_terms:
                handler.error(
                    f"Term {term.submitted} failed with matches {term.matched}"
                )
            handler.error("Search terms failed, see above for details.", abort=True)
            sys.exit(1)

        except httpx.RequestError as error:
            handler.error(str(error))
            handler.error(
                "BOLD Preprocess query failed, please check your network connection and try again.",
                abort=True,
            )
            sys.exit(1)
        finally:
            spinner.update(task, completed=1)


def get_stats(search_terms: str, handler: Handler) -> dict[str, int]:
    """Get some statistics about the query."""
    with handler.spin() as spinner:
        task = spinner.add_task(
            description="Getting speciment count from BOLD...", total=1
        )
        try:
            url = f"{CONFIG.apis.bold.host}/summary?query={search_terms}&fields=specimens,species"
            response = CLIENT.get(url).raise_for_status()
            spinner.update(
                task,
                description="Getting speciment count from BOLD...done.",
                completed=1,
            )
            return response.json().get("counts", {})
        except httpx.RequestError as error:
            handler.error(str(error))
            handler.error(
                "BOLD Summary query failed, please check your network connection and try again.",
                abort=True,
            )
            sys.exit(1)
        finally:
            spinner.update(task, completed=1)


def run_query(search_terms: str, output: Path, handler: Handler) -> None:
    """Request BOLD run the query and get the result ID."""
    with handler.spin() as spinner:
        task = spinner.add_task(description="Querying BOLD...", total=1)
        try:
            url = f"{CONFIG.apis.bold.host}/query?query={search_terms}&extend=full"
            response = CLIENT.get(url).raise_for_status()
            query_id = response.json().get("query_id")
            if query_id is None:
                handler.log(f"Response: {response.content.decode()}")
                handler.error(
                    "BOLD Query didn't return a query ID, instead it returned the above response.\nPlease try again or register an issue in the repository."
                )
                sys.exit(1)

            spinner.update(task, description="Querying BOLD...done.", completed=1)
            task = spinner.add_task(description="Downloading data...", total=1)

            url = f"{CONFIG.apis.bold.host}/documents/{query_id}/download?format=tsv"
            with CLIENT.stream("GET", url) as response, output.open("wb") as outfile:
                for data in response.iter_bytes():
                    outfile.write(data)

            spinner.update(task, description="Downloading data...done.", completed=1)

        except httpx.RequestError as error:
            handler.error(str(error))
            handler.error(
                "BOLD Summary query failed, please check your network connection and try again.",
                abort=True,
            )
            sys.exit(1)
        finally:
            spinner.update(task, completed=1)


def query_bold(geography: str, taxonomy: str, output: Path, handler: Handler) -> None:
    """Query BOLD public API for data that matches the search terms, writing to the given filepath."""
    term_matches = get_search_terms(geography, taxonomy, handler)
    selected_terms = select_terms(term_matches, handler)
    handler.log(f"Got terms: `{selected_terms}`")
    stats = get_stats(selected_terms, handler)
    handler.log(
        f"BOLD has {stats['specimens']} matching specimens representing {stats['species']} species."
    )
    if handler.confirm("Proceed with download?", abort=True):
        run_query(selected_terms, output, handler)
