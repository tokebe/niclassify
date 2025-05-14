from pathlib import Path
from typing import Callable
import requests
import shutil
from tempfile import NamedTemporaryFile
import os

from requests.compat import chardet
import httpx

from niclassify.core.interfaces import Handler

# TODO: upgrade to BOLDv5
# See https://boldsystems.org/data/api/#section1
# Basically, do a check with the preprocessor, get the middle term from the matches
# Or inform user of non-matches and attempt to recover
# Then, make a query and get back a request token
# Then, retrieve the data using the token

def query_bold(geography: str, taxonomy: str, output: Path, handler: Handler) -> None:
    api = "http://www.boldsystems.org/index.php/API_Public/combined?"

    request = api + "&".join([f"geo={geography}", f"taxon={taxonomy}", "format=tsv"])

    try:
        write_size = 0
        with (
            handler.spin() as spinner,
            open(output, "w", encoding="utf8") as file,
            httpx.Client(transport=httpx.HTTPTransport(retries=3), timeout=60, follow_redirects=True).stream(
                "GET",
                request,
            ) as response,
        ):
            task = spinner.add_task(description="Querying BOLD...", total=1)
            # error if response isn't success
            # TODO better error handling for this whole module
            try:
                response.raise_for_status()
            except Exception as error:
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
                try:
                    to_write = chunk.decode(encoding)
                except UnicodeDecodeError:
                    encoding = chardet.detect(chunk)["encoding"]
                    handler.debug(
                        f"Switched to encoding {encoding} after byte {write_size}"
                    )
                    to_write = chunk.decode(encoding)

                file.write(to_write)
                write_size += len(chunk)
                spinner.update(
                    task,
                    description=f"Querying BOLD...(received {write_size} bytes)",
                )

            task = spinner.update(
                task,
                description=f"Querying BOLD...done (wrote {write_size} bytes).",
                completed=1,
            )

            # handler.log("Success!")
            return

    except UnicodeDecodeError as error:
        handler.error(str(error))
        handler.error(handler.prefab.ERR_RESPONSE_DECODE, abort=True)

    except (
        OSError,
        IOError,
        KeyError,
        TypeError,
        ValueError,
        httpx.HTTPError,
    ) as error:
        handler.error(str(error))
        handler.error(handler.prefab.ERR_BOLD_SEARCH, abort=True)
