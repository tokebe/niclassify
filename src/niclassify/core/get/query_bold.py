from pathlib import Path

import chardet
import httpx

from niclassify.config.general import CONFIG
from niclassify.core.interfaces.handler import Handler

CLIENT = httpx.Client(
    transport=httpx.HTTPTransport(retries=3), timeout=120, follow_redirects=True
)

# TODO: The below will be useful eventually when bold v4 is cut off.
# Currently, it's not worth pursuing as bold v5 seems incredibly slow.
# See https://boldsystems.org/data/api/#section1

# class PreProcessorResponseTerm(BaseModel):
#     """A preprocessor search term result."""
#
#     submitted: str
#     matched: str
#
#
# class PreProcessorResponse(BaseModel):
#     """Response format to /api/query/preprocessor.
#
#     See https://portal.boldsystems.org/api/docs#/query/resolve_query_api_query_preprocessor_get
#     """
#
#     successful_terms: list[PreProcessorResponseTerm] | None = None
#     failed_terms: list[PreProcessorResponseTerm] | None = None
#
#
# def get_triplets(geography: str, taxonomy: str, handler: Handler) -> str:
#     """Refine user searchterms to a BOLD-approved query string."""
#     if geography.islower():
#         geography = geography.title()
#     if taxonomy.islower():
#         taxonomy = taxonomy.title()
#
#     with handler.spin() as spinner:
#         task = spinner.add_task(
#             description="Checking search terms with BOLD...", total=1
#         )
#
#         try:
#             final_terms = []
#             url = f"{CONFIG.apis.bold.host}/query/preprocessor?query=geo:{geography};tax:{taxonomy}"
#             response = CLIENT.get(url).raise_for_status()
#             handler.debug(str(response.json()))
#             body = PreProcessorResponse.model_validate(response.json())
#             if body.successful_terms is None:
#                 raise TypeError(
#                     "BOLD Preprocess query succeeded with no successful terms."
#                 )
#             for term in body.successful_terms:
#                 split_term = term.submitted.split(":")[0]
#                 prefix = split_term[0]
#                 name = split_term[-1]
#                 options = list[str]()
#                 for match in term.matched.split(";"):
#                     if prefix in match:
#                         options.append(match)
#                 if len(options) == 0:
#                     handler.error(
#                         f"No appropriate matches were found for term {name}. Try again, bearing in mind that BOLD is case-sensitive.",
#                         abort=True,
#                     )
#                 elif len(options) == 1:
#                     final_terms.append(options[0])
#                 else:
#                     handler.error(f"Got too many options: {options}", abort=True)
#             handler.debug(str(final_terms))
#             return ";".join(final_terms)
#
#         except httpx.HTTPStatusError as error:
#             body = PreProcessorResponse.model_validate(error.response.json())
#             if error.response.status_code != 400:
#                 handler.error(str(error))
#                 handler.error(
#                     f"BOLD Preprocess query returned with failing HTTP status {error.response.status_code}. See above for error details.",
#                     abort=True,
#                 )
#             if body.failed_terms is None:
#                 handler.error(str(error))
#                 handler.error(str(error.response))
#                 handler.error(
#                     "BOLD Preprocess query failed to process for unknown reasons, see above for error details.",
#                     abort=True,
#                 )
#                 sys.exit(1)
#
#             for term in body.failed_terms:
#                 handler.error(
#                     f"Term {term.submitted} failed with matches {term.matched}"
#                 )
#             handler.error("Search terms failed, see above for details.", abort=True)
#             sys.exit(1)
#
#         except httpx.RequestError as error:
#             handler.error(str(error))
#             handler.error(
#                 "BOLD Preprocess query failed, please check your network connection and try again.",
#                 abort=True,
#             )
#             sys.exit(1)
#         finally:
#             spinner.update(task, completed=1)
#


def query_bold(geography: str, taxonomy: str, output: Path, handler: Handler) -> None:
    """Query BOLD public API for data that matches the search terms, writing to the given filepath."""
    request = f"{CONFIG.apis.bold.host}/API_Public/combined?geo={geography}&taxon={taxonomy}&format=tsv"

    try:
        write_size = 0
        with (
            handler.spin() as spinner,
            output.open("w", encoding="utf8") as file,
            CLIENT.stream(
                "GET",
                request,
            ) as response,
        ):
            task = spinner.add_task(description="Querying BOLD...", total=1)
            # error if response isn't success
            # TODO better error handling for this whole module
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                handler.error(str(error))
                handler.error(handler.prefab.ERR_BOLD_SEARCH)
                return

            # Streamed result switches encoding occasionally for some reason
            encoding = None
            for chunk in response.iter_raw(chunk_size=int(1e6)):
                if not encoding:
                    encoding = chardet.detect(chunk)["encoding"]
                    handler.debug(f"Initial encoding {encoding}")
                to_write = ""
                if encoding is None:
                    raise TypeError("Encoding failed to be set.")
                try:
                    to_write = chunk.decode(encoding)
                except UnicodeDecodeError as e:
                    encoding = chardet.detect(chunk)["encoding"]
                    handler.debug(
                        f"Switched to encoding {encoding} after byte {write_size}"
                    )
                    if encoding is None:
                        raise TypeError("Encoding failed to be set.") from e
                    to_write = chunk.decode(encoding)

                file.write(to_write)
                write_size += len(chunk)
                spinner.update(
                    task,
                    description=f"Querying BOLD...(received {write_size} bytes)",
                )

            spinner.update(
                task,
                description=f"Querying BOLD...done (wrote {write_size} bytes).",
                completed=1,
            )

            # handler.log("Success!")
            return

    except UnicodeDecodeError as error:
        handler.error(str(error))
        handler.error(handler.prefab.ERR_RESPONSE_DECODE, abort=True)

    except (OSError, KeyError, TypeError, ValueError, httpx.HTTPError) as error:
        handler.error(str(error))
        handler.error(handler.prefab.ERR_BOLD_SEARCH, abort=True)
