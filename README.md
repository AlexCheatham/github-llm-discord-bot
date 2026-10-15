# Local LLM Discord Bot

A free Discord chatbot that runs an open-source model on your own PC through [Ollama](https://ollama.com). No paid APIs: Discord, Ollama and the models all cost nothing.

- @mention the bot in a channel, or DM it, and it replies.
- A system prompt gives it a personality, and a People list tells it who your friends are, so it calls them by name and can @ping them.
- Keyword **image triggers** reply with set pictures in code, so they fire every time instead of depending on the model.
- A **localhost-only web page** lets you edit all of this, pick a model and test-chat before anything goes to Discord.

## Setup

1. **Install Ollama** from https://ollama.com and pull a model:
   ```bash
   ollama pull llama3.2:3b
   ```
   On a 4 GB GPU, `llama3.2:3b` (fast) or `qwen3:4b` (smarter, slower) run fully on the GPU. Bigger models also work, but they spill over to the CPU and get slow.

2. **Install the Python packages** (Python 3.10+):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Try it locally first.** This needs no Discord setup:
   ```bash
   python bot.py --web-only
   ```
   On first run, the bot creates your own `config.json`, `system_prompt.txt` and `triggers.json` from the `*.example.*` files. These are gitignored, so your personal prompt and friends' usernames stay out of the repo.
   Open http://localhost:8080 to edit the system prompt, choose a model and test-chat.

4. **Create the Discord bot:**
   - Go to https://discord.com/developers/applications, click **New Application**, then **Bot**, then **Reset Token**, and copy the token.
   - On the same Bot page, turn on **Message Content Intent**.
   - Under **OAuth2 → URL Generator**, tick the scope `bot` and the permissions **Send Messages** and **Read Message History**. Open the generated URL and add the bot to your server.
   - Copy `.env.example` to `.env` and paste the token after `DISCORD_TOKEN=`. `.env` is gitignored, so the token never reaches GitHub.

5. **Run the bot.** This starts both the Discord bot and the web UI:
   ```bash
   python bot.py
   ```
   The bot is online while this window stays open.

## Usage

| Where | What |
|---|---|
| Discord | `@Bot what's a good name for a cat?` (or just DM it) |
| Discord | `@Bot !reset` clears that channel's conversation memory |
| Web UI | Edit the prompt, model, memory length, reply length, temperature and image triggers, then save. Changes apply immediately. |
| Web UI | Use **Chat as** to test how the bot treats a specific friend's username. |

## Image triggers

`triggers.json` is checked before the model runs. If a message contains one of the `request_words` (e.g. "pic", "draw"), the bot sends the first image whose `keywords` match. With no match, it sends the `fallback` line. `send` is either a link or a filename in an `images/` folder next to `bot.py`; `text` is optional.

Use links that don't expire (Imgur, Tenor, Giphy) or local files. Links copied from Discord attachments stop working after about a day.

## Files

| File | In git? | What it is |
|---|---|---|
| `bot.py` | yes | The bot, the Ollama calls and the web server |
| `web/index.html` | yes | The control panel page |
| `config.example.json` → `config.json` | example only | Model, memory length (0 = none), reply length, temperature, repeat penalty, web port |
| `system_prompt.example.txt` → `system_prompt.txt` | example only | Personality and People list. Re-read on every message, so edits apply without a restart. |
| `triggers.example.json` → `triggers.json` | example only | Image keyword triggers |
| `.env.example` → `.env` | example only | Your Discord bot token |
| `images/` | no | Local images used by triggers |

People lines in the system prompt must look like `- @discordusername = Name. Anything else.` The bot reads them to show names instead of usernames and to turn `@Name` in replies into real pings.

Conversation memory is kept in RAM only, so it resets when the bot restarts. The web UI listens on `127.0.0.1` only and can't be reached from other machines.
