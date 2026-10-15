# Usage

## Summarizing a repository
Run repobrief run owner/name. The summary is written to the reports folder.
Use --out to choose another folder.

## Output formats
The default format is markdown. Use --format json or --format csv for other formats.

## Web viewer
repobrief serve starts a small web page at http://localhost:8000 that lists your reports.

## Running on a schedule
repobrief has no built-in scheduler. Use cron on Linux and macOS, or Task Scheduler on
Windows, to run repobrief run at the time you want.
