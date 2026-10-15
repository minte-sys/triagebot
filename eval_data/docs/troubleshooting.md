# Troubleshooting

## GitHub rate limits
Without a token, GitHub allows 60 requests per hour and large repositories fail with
HTTP 403 rate limit exceeded. Set GITHUB_TOKEN in .env to get 5000 requests per hour.

## Model timeouts
Very long READMEs can make the model slower than timeout_s. Raise timeout_s in the [llm]
section of config.toml.

## Unicode errors on Windows
If the console shows UnicodeEncodeError, set the environment variable PYTHONUTF8=1.

## Private repositories
For private repositories the token needs read access to the repository's contents.
Without it GitHub answers 404 Not Found.
