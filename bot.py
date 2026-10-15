"""Free local LLM Discord bot.

Runs an open-source model through Ollama on this PC, answers @mentions and DMs
on Discord, and serves a localhost-only web page for editing the system prompt
and testing the model.

    python bot.py             # Discord bot + web UI
    python bot.py --web-only  # web UI only (no Discord token needed)
"""

import argparse
import asyncio
import json
import os
import re
import shutil
from pathlib import Path

import discord
import ollama
from aiohttp import web
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
PROMPT_PATH = ROOT / "system_prompt.txt"
WEB_DIR = ROOT / "web"

DEFAULT_CONFIG = {
    "model": "llama3.2:3b",
    "history_length": 10,
    "max_reply_tokens": 150,
    "temperature": 0.6,      # lower = calmer, more on-script; higher = more random
    "repeat_penalty": 1.1,   # >1 discourages looping the same phrases
    "web_port": 8080,
}
DISCORD_LIMIT = 2000


# ---------------------------------------------------------------- config / prompt

def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""


def save_prompt(text: str) -> None:
    PROMPT_PATH.write_text(text.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------- generation

ollama_client = ollama.AsyncClient()
gpu_lock = asyncio.Lock()  # one generation at a time; the GPU is small
histories: dict[str, list[dict]] = {}  # conversation key -> chat turns


class GenerationError(Exception):
    pass


async def generate(key: str, user_text: str) -> str:
    """Add user_text to the conversation `key`, ask the model, return its reply."""
    config = load_config()
    history = histories.setdefault(key, [])
    history.append({"role": "user", "content": user_text})

    messages = history[-max(config["history_length"], 1):]  # always includes the new message
    system = load_prompt()  # re-read every time, so edits apply immediately
    if system:
        messages = [{"role": "system", "content": system}] + messages

    # Stop if the model starts writing other people's lines ("\nSam: ...").
    names = {"User"} | {name.split()[0] for name in load_people().values() if name}
    options = {
        "num_predict": config["max_reply_tokens"],
        "temperature": config["temperature"],
        "repeat_penalty": config["repeat_penalty"],
        "stop": [f"\n{name}:" for name in sorted(names)],
    }
    if config.get("num_ctx"):  # optional: context window in tokens (Ollama's default otherwise)
        options["num_ctx"] = config["num_ctx"]

    try:
        async with gpu_lock:
            response = await ollama_client.chat(model=config["model"], messages=messages, options=options)
    except ollama.ResponseError as e:
        history.pop()
        if e.status_code == 404:
            raise GenerationError(
                f"Model `{config['model']}` isn't installed. Run: ollama pull {config['model']}"
            ) from e
        raise GenerationError(f"Ollama error: {e.error}") from e
    except (ConnectionError, OSError) as e:
        history.pop()
        raise GenerationError("Can't reach Ollama. Is it running? (start the Ollama app)") from e

    content = re.sub(r"<think>.*?</think>", "", response.message.content or "", flags=re.DOTALL)  # thinking models
    content = content.strip()
    if response.done_reason == "length":
        content = trim_to_last_sentence(content)
    reply = content or "(empty reply)"
    history.append({"role": "assistant", "content": reply})
    keep_last(history, config["history_length"])
    return reply


# ---------------------------------------------------------------- image triggers
# Keyword rules checked in code before the model runs, so they always fire the same way.

TRIGGERS_PATH = ROOT / "triggers.json"
IMAGES_DIR = ROOT / "images"


def load_triggers() -> dict:
    if TRIGGERS_PATH.exists():
        return json.loads(TRIGGERS_PATH.read_text(encoding="utf-8"))
    return {"request_words": [], "images": [], "fallback": ""}


def has_word(text: str, words: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(w.lower())}(?!\w)", text) for w in words if w.strip())


def check_triggers(text: str) -> tuple[str, list[Path]] | None:
    """If the message asks for an image, return (text, files) to send instead of asking the model."""
    triggers = load_triggers()
    lowered = text.lower()
    if not has_word(lowered, triggers.get("request_words", [])):
        return None
    for image in triggers.get("images", []):
        if has_word(lowered, image.get("keywords", [])):
            return resolve_image(image.get("send", ""), image.get("text", ""))
    return triggers.get("fallback", ""), []


def resolve_image(target: str, text: str = "") -> tuple[str, list[Path]]:
    """A link goes in the message text (Discord embeds it); a file in images/ is uploaded."""
    if target.startswith(("http://", "https://")):
        return f"{text}\n{target}".strip(), []
    path = (IMAGES_DIR / target).resolve()
    if target and path.parent == IMAGES_DIR.resolve() and path.is_file():
        return text, [path]
    return f"{text}\n(missing image: {target})".strip(), []


def remember_exchange(key: str, user_text: str, reply: str) -> None:
    """Record a reply the bot sent without the model, so later replies have the context."""
    history = histories.setdefault(key, [])
    history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
    keep_last(history, load_config()["history_length"])


def keep_last(history: list, n: int) -> None:
    """Trim history in place to its last n turns (0 = remember nothing)."""
    del history[:max(len(history) - n, 0)]


def trim_to_last_sentence(text: str) -> str:
    """Cut a reply that hit the length cap back to its last complete sentence or line."""
    cut = max(text.rfind(c) for c in ".!?\n")
    return text[:cut + 1].strip() if cut > 0 else text


# "- @username = Name. anything else" lines in the system prompt's People list.
PEOPLE_LINE = re.compile(r"^\s*-\s*@([\w.]+)\s*=\s*([^.\n]+)", re.MULTILINE)


def load_people() -> dict[str, str]:
    """Discord username -> name to call them, read from the system prompt."""
    return {user.lower(): name.strip() for user, name in PEOPLE_LINE.findall(load_prompt())}


def person_name(username: str, fallback: str) -> str:
    return load_people().get(username.lower(), fallback)


def speaker_prefix(display_name: str, username: str) -> str:
    """'Sam: ' for people in the People list, otherwise 'Nick (@username): '."""
    name = load_people().get(username.lower())
    return f"{name}: " if name else f"{display_name} (@{username}): "


known_ids: dict[str, int] = {}  # username (lowercase) -> Discord user ID, learned as people talk
PING_TOKEN = re.compile(r"(?<![\w@])@([\w.]*\w)")
NO_MASS_PINGS = discord.AllowedMentions(everyone=False, roles=False, users=True, replied_user=False)


def remember(user: discord.abc.User) -> None:
    known_ids[user.name.lower()] = user.id


async def find_user_id(guild: discord.Guild | None, username: str) -> int | None:
    if username in known_ids:
        return known_ids[username]
    if guild is None:
        return None
    try:
        members = await guild.query_members(query=username, limit=5)
    except (discord.HTTPException, discord.ClientException, asyncio.TimeoutError):
        return None
    for member in members:
        remember(member)
    return known_ids.get(username)


async def add_pings(text: str, guild: discord.Guild | None) -> str:
    """Turn '@Sam' / '@sam_123' in the model's reply into real Discord pings."""
    people = load_people()
    by_name = {name.split()[0].lower(): user for user, name in people.items() if name}
    ids = {}
    for token in set(PING_TOKEN.findall(text)):
        username = token.lower() if token.lower() in people else by_name.get(token.lower())
        if username and (user_id := await find_user_id(guild, username)):
            ids[token] = user_id
    return PING_TOKEN.sub(lambda m: f"<@{ids[m.group(1)]}>" if m.group(1) in ids else m.group(0), text)


def split_for_discord(text: str) -> list[str]:
    chunks = []
    while len(text) > DISCORD_LIMIT:
        cut = text.rfind("\n", 0, DISCORD_LIMIT)
        if cut <= 0:
            cut = DISCORD_LIMIT
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    chunks.append(text)
    return chunks


# ---------------------------------------------------------------- discord

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


@bot.event
async def on_ready():
    print(f"Discord: logged in as {bot.user} (model: {load_config()['model']})")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    is_dm = isinstance(message.channel, discord.DMChannel)
    # Pinging the bot's auto-created role (same name as the bot) counts as pinging the bot.
    my_roles = [r for r in message.role_mentions if r.is_bot_managed() and r in message.guild.me.roles] if message.guild else []
    if not is_dm and bot.user not in message.mentions and not my_roles:
        return

    text = re.sub(rf"<@!?{bot.user.id}>", "", message.content)
    for role in my_roles:
        text = text.replace(f"<@&{role.id}>", "")
    text = text.strip()
    remember(message.author)
    # Turn other people's pings (<@1234>) into names so the model knows who's meant.
    for user in message.mentions:
        remember(user)
        text = re.sub(rf"<@!?{user.id}>", "@" + person_name(user.name, user.name), text)
    key = f"discord:{message.channel.id}"

    if text.lower() == "!reset":
        histories.pop(key, None)
        await message.reply("Conversation history cleared.")
        return
    if not text:
        await message.reply("Mention me with a message and I'll reply.")
        return

    user_text = speaker_prefix(message.author.display_name, message.author.name) + text
    image_files: list[Path] = []
    if (triggered := check_triggers(text)) is not None:
        reply, image_files = triggered
        remember_exchange(key, user_text, reply or "[sent an image]")
    else:
        async with message.channel.typing():
            try:
                reply = await generate(key, user_text)
            except GenerationError as e:
                reply = f"⚠️ {e}"
            reply = await add_pings(reply, message.guild)

    chunks = split_for_discord(reply)
    files = [discord.File(path) for path in image_files]
    if not chunks[0] and not files:
        chunks[0] = "*shrugs*"
    await message.reply(chunks[0] or None, files=files, allowed_mentions=NO_MASS_PINGS)
    for chunk in chunks[1:]:
        await message.channel.send(chunk, allowed_mentions=NO_MASS_PINGS)


# ---------------------------------------------------------------- web UI

@web.middleware
async def local_only(request: web.Request, handler):
    # Block DNS-rebinding and cross-site requests from other web pages.
    host = request.host.split(":")[0]
    if host not in ("localhost", "127.0.0.1"):
        raise web.HTTPForbidden(text="localhost only")
    if request.method == "POST" and request.content_type != "application/json":
        raise web.HTTPUnsupportedMediaType(text="expected application/json")
    return await handler(request)


async def index(request):
    return web.FileResponse(WEB_DIR / "index.html")


def prompt_version() -> str:
    return str(PROMPT_PATH.stat().st_mtime_ns) if PROMPT_PATH.exists() else ""


async def get_prompt(request):
    return web.json_response({"prompt": load_prompt(), "version": prompt_version()})


async def post_prompt(request):
    data = await request.json()
    # Refuse to overwrite edits made elsewhere (another tab, a text editor) since this page loaded.
    if not data.get("force") and data.get("version") != prompt_version():
        return web.json_response(
            {"error": "system_prompt.txt was changed somewhere else since this page loaded. "
                      "Copy your edits, reload the page, and paste them back in."},
            status=409,
        )
    save_prompt(str(data.get("prompt", "")))
    return web.json_response({"ok": True, "version": prompt_version()})


async def get_models(request):
    try:
        listed = await ollama_client.list()
        names = sorted(m.model for m in listed.models)
    except (ollama.ResponseError, ConnectionError, OSError):
        return web.json_response({"models": [], "error": "Can't reach Ollama. Is it running?"})
    return web.json_response({"models": names})


async def get_config(request):
    return web.json_response(load_config())


async def post_config(request):
    data = await request.json()
    config = load_config()
    try:
        if "model" in data:
            config["model"] = str(data["model"]).strip()
        for field, low in (("history_length", 0), ("max_reply_tokens", 1)):
            if field in data:
                value = int(data[field])
                if value < low:
                    raise ValueError
                config[field] = value
        for field, low, high in (("temperature", 0, 2), ("repeat_penalty", 0.5, 2)):
            if field in data:
                value = float(data[field])
                if not low <= value <= high:
                    raise ValueError
                config[field] = value
    except (TypeError, ValueError):
        raise web.HTTPBadRequest(
            text="history and max tokens must be positive whole numbers; "
                 "temperature must be 0-2 and repeat penalty 0.5-2"
        )
    save_config(config)
    return web.json_response(config)


async def post_chat(request):
    data = await request.json()
    text = str(data.get("message", "")).strip()
    if not text:
        raise web.HTTPBadRequest(text="message is empty")
    username = str(data.get("as", "")).strip().lstrip("@") or "tester"
    user_text = speaker_prefix(username, username) + text
    if (triggered := check_triggers(text)) is not None:
        reply, files = triggered
        remember_exchange("web", user_text, reply or "[sent an image]")
        reply = "\n".join([reply] + [f"[attached image: {path.name}]" for path in files]).strip()
        return web.json_response({"reply": reply})
    try:
        reply = await generate("web", user_text)
    except GenerationError as e:
        return web.json_response({"error": str(e)}, status=502)
    return web.json_response({"reply": reply})


async def get_triggers(request):
    text = TRIGGERS_PATH.read_text(encoding="utf-8") if TRIGGERS_PATH.exists() else "{}"
    return web.json_response({"triggers": text})


async def post_triggers(request):
    data = await request.json()
    text = str(data.get("triggers", ""))
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("images", []), list):
            raise ValueError("expected an object with an \"images\" list")
    except ValueError as e:
        return web.json_response({"error": f"Triggers aren't valid JSON: {e}"}, status=400)
    TRIGGERS_PATH.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return web.json_response({"ok": True})


async def post_reset(request):
    histories.pop("web", None)
    return web.json_response({"ok": True})


def make_web_app() -> web.Application:
    app = web.Application(middlewares=[local_only])
    app.add_routes([
        web.get("/", index),
        web.get("/api/prompt", get_prompt),
        web.post("/api/prompt", post_prompt),
        web.get("/api/models", get_models),
        web.get("/api/config", get_config),
        web.post("/api/config", post_config),
        web.post("/api/chat", post_chat),
        web.get("/api/triggers", get_triggers),
        web.post("/api/triggers", post_triggers),
        web.post("/api/reset", post_reset),
    ])
    return app


# ---------------------------------------------------------------- main

def create_local_files() -> None:
    """On a fresh clone, start the gitignored personal files from the public examples."""
    for path in (CONFIG_PATH, PROMPT_PATH, TRIGGERS_PATH):
        example = path.with_name(path.stem + ".example" + path.suffix)
        if not path.exists() and example.exists():
            shutil.copyfile(example, path)
            print(f"Created {path.name} from {example.name}. Edit it to make the bot your own.")


async def main(web_only: bool) -> None:
    create_local_files()
    load_dotenv(ROOT / ".env")
    port = load_config()["web_port"]

    runner = web.AppRunner(make_web_app())
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    print(f"Web UI: http://localhost:{port}")

    try:
        if web_only:
            print("Web-only mode (Discord disabled). Ctrl+C to stop.")
            await asyncio.Event().wait()
        else:
            token = os.getenv("DISCORD_TOKEN")
            if not token:
                raise SystemExit("DISCORD_TOKEN is not set. Copy .env.example to .env and add your bot token.")
            async with bot:
                await bot.start(token)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--web-only", action="store_true", help="run only the local web UI")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.web_only))
    except KeyboardInterrupt:
        pass
