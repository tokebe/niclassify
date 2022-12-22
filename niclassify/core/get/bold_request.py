from pathlib import Path
from typing import Callable
import requests
import shutil


def bold_request(geography: str, taxonomy: str, output: Path, handler: Callable):
    api = "http://www.boldsystems.org/index.php/API_Public/combined?"

    request = api + "&".join([f"geo={geography}", f"taxon={taxonomy}", "format=tsv"])

    try:
        attempts = 3
        while True:
            if attempts == 0:
                handler.error(TimeoutError("site keeps timing out"))
                break
            handler.log("Making request...")
            try:
                with open(output, "wb") as file, requests.get(
                    request, stream=True
                ) as response:

                    # error if response isn't success
                    # TODO better handling?
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        handler.error(e)
                        handler.error(requests.RequestException(handler.prefab.BOLD_SEARCH_ERR))
                        return
                    shutil.copyfileobj(response.raw, file)

                handler.log("Success!")
                return
            except requests.exceptions.Timeout:
                attempts -= 1
                handler.log(
                    "    request timed out, trying again ({} of 3)...".format(
                        3 - attempts
                    )
                )
                pass
            except requests.exceptions.RequestException as e:
                handler.error(e)
                handler.error(requests.exceptions.RequestException(handler.prefab.BOLD_SEARCH_ERR))
                break

    except UnicodeDecodeError as e:
        handler.error(e)
        handler.error(Exception(handler.prefab.RESPONSE_DECODE_ERR))

    except requests.RequestException:
        handler.error(e)
        handler.error(requests.RequestException(handler.prefab.BOLD_SEARCH_ERR))

    except (OSError, IOError, KeyError, TypeError, ValueError) as e:
        handler.error(e)
        handler.error(OSError(handler.prefab.BOLD_SEARCH_ERR))
