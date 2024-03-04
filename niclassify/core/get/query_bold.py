from pathlib import Path
from typing import Callable
import requests
import shutil
from tempfile import NamedTemporaryFile
import os

from requests.compat import chardet

from ..interfaces import Handler


def query_bold(geography: str, taxonomy: str, output: Path, handler: Handler) -> None:
    api = "http://www.boldsystems.org/index.php/API_Public/combined?"

    request = api + "&".join([f"geo={geography}", f"taxon={taxonomy}", "format=tsv"])

    try:
        attempts = 3
        while True:
            if attempts == 0:
                handler.error(TimeoutError("boldsystems.com keeps timing out"))
                break
            try:
                write_size = 0
                with handler.spin() as spinner:
                    task = spinner.add_task(description="Querying BOLD...", total=1)
                    with open(output, "w", encoding="utf8") as file, requests.get(
                        request, stream=True
                    ) as response:
                        # error if response isn't success
                        # TODO better error handling for this whole module
                        try:
                            response.raise_for_status()
                        except Exception as error:
                            handler.error(error)
                            handler.error(
                                requests.RequestException(
                                    handler.prefab.BOLD_SEARCH_ERR
                                )
                            )
                            return

                        # Streamed result switches encoding occasionally for some reason
                        encoding = None
                        for chunk in response.iter_content(chunk_size=int(1e6)):
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
            except requests.exceptions.Timeout:
                attempts -= 1
                handler.debug(
                    "    request timed out, trying again ({} of 3)...".format(
                        3 - attempts
                    )
                )
                pass
            except requests.exceptions.RequestException as error:
                with handler.debug_lock:
                    handler.debug(error)
                    handler.error(
                        requests.exceptions.RequestException(
                            handler.prefab.BOLD_SEARCH_ERR
                        )
                    )
                break

    except UnicodeDecodeError as error:
        handler.error(error)
        handler.error(Exception(handler.prefab.RESPONSE_DECODE_ERR), abort=True)

    except requests.RequestException as error:
        handler.error(error)
        handler.error(
            requests.RequestException(handler.prefab.BOLD_SEARCH_ERR), abort=True
        )

    except (OSError, IOError, KeyError, TypeError, ValueError) as error:
        handler.error(error)
        handler.error(OSError(handler.prefab.BOLD_SEARCH_ERR), abort=True)
