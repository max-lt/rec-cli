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

To make it permanent, add this line to your `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export MISTRAL_API_KEY=your_key_here' >> ~/.zshrc  # or ~/.bashrc
```

## Usage

```bash
rec              # Record → Enter → transcription to stdout
rec -c           # Same, but also copy to clipboard
rec --clip       # Same as -c
rec -f audio.wav # Transcribe an existing audio file
rec --file audio.wav # Same as -f
```

### Pipe it

```bash
rec | pbcopy                    # macOS: copy to clipboard
rec | xclip -selection clip     # Linux: copy to clipboard
rec >> notes.txt                # Append to file
echo "$(rec)" | some-command    # Use in scripts
```

## How it works

1. Starts recording from default microphone
2. Press Enter to stop
3. Sends audio to [Voxtral](https://mistral.ai/news/voxtral) (Mistral's speech-to-text API)
4. Prints transcription to stdout

Status messages (`Recording...`, `2.1s transcribing...`) go to stderr, so they don't interfere with piping.

## License

MIT
