# rec

Quick speech-to-text for devs. Record audio, press Enter, get text.

```
$ rec
Recording...
2.1s transcribing...
Hello, this is a test.
```

## Install

```bash
cargo install --git https://github.com/max-lt/rec-cli
```

Or from source:

```bash
git clone https://github.com/max-lt/rec-cli
cargo install --path rec-cli
```

Requires a [Mistral API key](https://console.mistral.ai/):

```bash
export MISTRAL_API_KEY=your_key_here
```

For Claude correction (optional), also set:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

To make it permanent, add these lines to your `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export MISTRAL_API_KEY=your_key_here' >> ~/.zshrc  # or ~/.bashrc
echo 'export ANTHROPIC_API_KEY=your_key_here' >> ~/.zshrc  # optional
```

## Usage

### Basic transcription

```bash
rec              # Record → Enter → transcription to stdout
rec -c           # Same, but also copy to clipboard
rec --clip       # Same as -c
rec -f audio.wav # Transcribe an existing audio file
rec --file audio.wav # Same as -f
```

### Claude correction

Improve transcription accuracy with Claude AI (requires `ANTHROPIC_API_KEY`):

```bash
rec --correct              # Correct transcription using Claude
rec --correct --clip       # Correct and copy to clipboard (only corrected version)
rec -f audio.wav --correct # Correct transcription from file
rec --correct --debug      # Show Claude's correction details
```

### Custom vocabulary

Add technical terms or proper nouns that Claude should recognize:

```bash
rec add-word Anthropic
rec add-word Voxtral
rec add-word OpenWorkers
```

Words are stored in config file (see Configuration below).

## Configuration

Config file location (auto-created on first use):
- macOS: `~/Library/Application Support/rec/config.json`
- Linux: `~/.config/rec/config.json`

Example configuration:

```json
{
  "custom_words": [
    "Anthropic",
    "Voxtral",
    "OpenWorkers"
  ],
  "claude_model": "claude-haiku-4-5"
}
```

You can change the Claude model to use different models like `claude-sonnet-4-5` for better quality.

### Correction History

When using `--correct`, both the original and corrected transcriptions are saved to a history file:
- macOS: `~/Library/Application Support/rec/history.json`
- Linux: `~/.config/rec/history.json`

The output will show both versions (original in gray, corrected in normal color):
```
Ils font vraiment du bon travail en tropique.  (dimmed/gray)

Ils font vraiment du bon travail à Anthropic.  (normal)
```

With `--clip`, only the final corrected text is copied to clipboard. Use `--debug` to see detailed correction information from Claude.

History entries include timestamp, both versions, model used, and custom words that were active. This data can be useful for:
- Training ML models
- Analyzing correction patterns
- Providing context to Claude for better future corrections (last 5 entries are used as context)

### Pipe it

```bash
rec | pbcopy                    # macOS: copy to clipboard
rec | xclip -selection clip     # Linux: copy to clipboard
rec >> notes.txt                # Append to file
echo "$(rec)" | some-command    # Use in scripts
```

## How it works

### Basic transcription

1. Starts recording from default microphone
2. Press Enter to stop
3. Sends audio to [Voxtral](https://mistral.ai/news/voxtral) (Mistral's speech-to-text API)
4. Prints transcription to stdout

### With Claude correction (`--correct`)

1. Transcribes with Mistral Voxtral (same as above)
2. Sends transcription + custom words to Claude API
3. Claude corrects obvious errors and applies custom vocabulary
4. Prints corrected transcription

Status messages (`Recording...`, `2.1s transcribing...`, `Correcting with Claude...`) go to stderr, so they don't interfere with piping.

**Note**: Mistral's Voxtral API does not support vocabulary hints, so custom words are only used by Claude for post-correction.

### Using Rec API

Alternatively, you can use [Rec API](https://rec-api.workers.rocks) instead of calling Mistral directly. Your audio data may be used for research purposes.

```bash
export REC_API_URL=https://rec-api.workers.rocks
export REC_API_KEY=rec_your_token_here
```

When both `REC_API_URL` and `REC_API_KEY` are set, `rec` will use the API automatically.

## License

MIT
