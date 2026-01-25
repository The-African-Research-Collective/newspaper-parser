# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "africanlanguages",
#     "datasets",
#     "datatrove",
#     "huggingface_hub",
#     "surt",
#     "tenacity",
#     "typer",
#     "warcio",
# ]
#
# [tool.uv.sources]
# africanlanguages = { git = "https://github.com/The-African-Research-Collective/africanlanguages.git", rev = "b37d787c8d5cf8fac0571dac87e8214cb9e4716f" }
# ///
# Usage: uv run scripts/finepdf_download.py download-fineweb-african-pdfs --download-dir data/finepdfs -e dag_Latn -e pcm_Latn
# Notes:
#   - We ignore dag_Latn and pcm_Latn because the eye test shows most of the PDFs are not the correct language
#   - Together, they consititure 1.9M out of the 2.0M PDFs listed as African languages in FinePDF
#   - We're crawling politely as CommonCrawl suggests here: https://commoncrawl.org/blog/oct-nov-2023-performance-issues
from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import uuid
from functools import lru_cache, partial, wraps
from io import BytesIO
from typing import Annotated, Any, Callable

import aiohttp
import pyarrow.parquet as pq
from africanlanguages import AfricanLanguages
from datasets import get_dataset_config_names, load_dataset, Dataset
from huggingface_hub import snapshot_download
from rich import print as rich_print
from surt import surt
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from typer import Option, Typer
from warcio.archiveiterator import ArchiveIterator

# NOTE: This limits the number of downloads that happens at once
download_semaphore = asyncio.Semaphore(10)


class AsyncTyper(Typer):
    @staticmethod
    def maybe_run_async(decorator: Callable, func: Callable) -> Any:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            def runner(*args: Any, **kwargs: Any) -> Any:
                return asyncio.run(func(*args, **kwargs))

            decorator(runner)
        else:
            decorator(func)
        return func

    def callback(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)


app = AsyncTyper(no_args_is_help=True)


def deterministic_id(string: str) -> str:
    return str(uuid.UUID(hashlib.md5(string).hexdigest()))


@retry(
    stop=stop_after_attempt(1_000),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, OSError)),
)
def query_commoncrawl_parquet(crawl_filepath: str, surt_key: str) -> dict | None:
    pf = pq.ParquetFile(crawl_filepath)

    for i in range(pf.num_row_groups):
        rg = pf.metadata.row_group(i)
        col_idx = pf.schema_arrow.get_field_index("url_surtkey")
        stats = rg.column(col_idx).statistics

        if (
            not stats
            or stats.min is None
            or stats.max is None
            or not (stats.min <= surt_key <= stats.max)
        ):
            continue

        df = pf.read_row_groups(
            [i],
            columns=[
                "url_surtkey",
                "url",
                "warc_filename",
                "warc_record_offset",
                "warc_record_length",
            ],
        ).to_pandas()

        match = df[df["url_surtkey"] == surt_key]
        if len(match) > 0:
            return match.iloc[0].to_dict()

    return None


@retry(
    stop=stop_after_attempt(1_000),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, OSError)),
)
async def get_record_length_from_warc_headers(
    session: aiohttp.ClientSession, warc_filename: str, offset: str
) -> int:
    chunk_size = 1024 * 1024
    url = f"https://data.commoncrawl.org/{warc_filename}"
    headers = {"Range": f"bytes={offset}-{offset + chunk_size - 1}"}

    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        content = await resp.read()

    buffer = BytesIO(content)
    record = next(ArchiveIterator(buffer))
    return int(record.rec_headers.get_header("Content-Length"))


@retry(
    stop=stop_after_attempt(1_000),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, OSError)),
)
async def download_warc_record(
    session: aiohttp.ClientSession,
    download_path: str,
    warc_filename: str,
    offset: int,
    length: int = None,
) -> None:
    """
    Download a single WARC record from Common Crawl asynchronously.
    Returns the content of the record as bytes.
    """
    url = f"https://data.commoncrawl.org/{warc_filename}"

    if length is None:
        length = await get_record_length_from_warc_headers(
            session, warc_filename, offset
        )

    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        content = await resp.read()

    # Extract the record from the WARC content
    record = next(ArchiveIterator(BytesIO(content)))
    with open(download_path, "wb") as f:
        f.write(record.content_stream().read())


async def download_pdf(
    row: dict, download_dir: str, session: aiohttp.ClientSession
) -> dict:
    """
    Full async function: get WARC content for a given URL.
    """
    os.makedirs(f"{download_dir}/{row['language']}", exist_ok=True)
    filename = f"{deterministic_id(row['id'].encode('utf-8'))}.pdf"
    download_path = f"{download_dir}/{row['language']}/{filename}"

    row["file_name"] = filename

    if os.path.exists(download_path):
        return row

    warc_filename = row["file_path"].replace("s3://commoncrawl/", "")
    offset = int(row["offset"])
    length = None

    async with download_semaphore:
        if warc_filename.startswith("cc-index"):
            cdx_results = query_commoncrawl_parquet(row["file_path"], surt(row["url"]))

            if cdx_results:
                warc_filename = cdx_results["warc_filename"]
                # Because we select the first warc_file that has a url match, there's a chance the offset is different
                # HF may have collected from a different warc_file
                offset = cdx_results["warc_record_offset"]
                length = int(cdx_results["warc_record_length"])
            else:
                row["file_name"] = None

        if warc_filename.startswith("crawl-data"):
            await download_warc_record(
                warc_filename=warc_filename,
                offset=offset,
                length=length,
                download_path=download_path,
                session=session,
            )
    return row


@app.command()
@lru_cache()
def list_african_languages_in_finepdf(
    verbose: Annotated[
        bool, Option(help="If verbose, print african languages")
    ] = False,
) -> list[str]:
    configs = get_dataset_config_names("HuggingFaceFW/finepdfs")
    finepdf_african_languages = [
        language
        for language in configs
        if language.split("_")[0].upper() in AfricanLanguages._member_names_
    ]

    if verbose:
        rich_print(finepdf_african_languages)

    return finepdf_african_languages


@app.command()
def download_hf_fineweb_african_data(
    download_dir: Annotated[str, Option(help="Directory to download data to")],
    include_languages: Annotated[
        str, Option("-i", "--include-language", help="Languages to include")
    ] = None,
    exclude_languages: Annotated[
        list[str] | None,
        Option("-e", "--exclude-language", help="language codes to exclude"),
    ] = None,
):
    snapshot_download(
        repo_id="HuggingFaceFW/finepdfs",
        repo_type="dataset",
        allow_patterns=[
            f"data/{language}/**/*.parquet"
            for language in list_african_languages_in_finepdf()
            if (
                # Either exclude_languages is Falsy or language is not in exclude_languages
                (not exclude_languages or language not in exclude_languages)
                # Either include_languages is Falsy or language is in include_languages
                and (not include_languages or language in include_languages)
            )
        ],
        local_dir=download_dir,
    )


@app.command()
async def download_fineweb_african_pdfs(
    download_dir: Annotated[str, Option(help="Directory to download data to")],
    include_languages: Annotated[
        str, Option("-i", "--include-language", help="Languages to include")
    ] = None,
    exclude_languages: Annotated[
        list[str] | None,
        Option("-e", "--exclude-language", help="language codes to exclude"),
    ] = None,
):
    PDF_DOWNLOAD_FOLDER = os.path.join(download_dir, "pdfs")

    download_hf_fineweb_african_data(
        download_dir=download_dir,
        include_languages=include_languages,
        exclude_languages=exclude_languages,
    )

    ds: dict[str, Dataset] = load_dataset(
        "parquet",
        data_files={
            language: f"{download_dir}/data/{language}/*/*.parquet"
            for language in list_african_languages_in_finepdf()
            if (
                # Either exclude_languages is Falsy or language is in exclude_languages
                (not exclude_languages or language not in exclude_languages)
                # Either include_languages is Falsy or language is in include_languages
                and (not include_languages or language in include_languages)
            )
        },
    )

    async with aiohttp.ClientSession() as session:
        for _, language_ds in ds.items():
            for batch in language_ds.batch(batch_size=1_000):
                tasks = [
                    download_pdf(row, PDF_DOWNLOAD_FOLDER, session=session)
                    for row in Dataset.from_dict(batch)
                    if not os.path.exists(
                        os.path.join(
                            PDF_DOWNLOAD_FOLDER,
                            row["language"],
                            f"{deterministic_id(row['id'].encode('utf-8'))}.pdf",
                        )
                    )
                ]
                if not tasks:
                    continue

                await asyncio.gather(*tasks)

    # TODO: Do we rewrite the dataset to file?


if __name__ == "__main__":
    app()
