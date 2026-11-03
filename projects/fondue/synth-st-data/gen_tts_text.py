"""Stream E, step 1: English utterances for TTS, sampled from an instruct LLM.

One prompt per (domain, scenario, register, entity seed). Each asks for `per_prompt`
utterances, one per line, rich in names, places, dates, numbers and medication/product
names; the entity seeds are random picks from the pools below so that prompts differ.
Domains cover service encounters plus the intended use cases (online video, film and
series, voice messages); places are drawn from all over Europe.
Output: JSONL rows {domain, scenario, register, text, prompt_id}. The default model is
EuroLLM-9B-Instruct (synth.sif, vLLM 0.14); Qwen3.6-35B-A3B needs vLLM >= 0.19, see
setup_qwen_venv.sh and slurm/gen_tts_text_artemis.sbatch. Thinking is switched off.
Deduplicate afterwards with dedup_text.py.
"""

import random
import re
import time
from pathlib import Path

import tyro

from common import MT_MODEL, ledger_append, write_jsonl_atomic


DOMAINS = {
    "healthcare": {
        "doctor-patient consultation": ["a family doctor", "a patient"],
        "pharmacy counter": ["a pharmacist", "a customer"],
        "emergency call": ["an emergency dispatcher", "a caller"],
        "hospital admission and ward round": ["a nurse", "a patient or relative"],
        "booking a medical appointment": ["a receptionist", "a caller"],
    },
    "travel": {
        "airport check-in and boarding": ["a ground-staff member", "a traveller"],
        "train and metro announcements": ["a station announcer"],
        "hotel reception": ["a receptionist", "a guest"],
        "asking for directions": ["a pedestrian", "a local"],
        "restaurant ordering": ["a waiter", "a diner"],
    },
    "customer service and banking": {
        "bank branch or phone banking": ["a bank clerk", "a customer"],
        "telecom or utility support call": ["a support agent", "a customer"],
        "online order complaint": ["a support agent", "a customer"],
        "insurance claim": ["a claims handler", "a policyholder"],
    },
    "public administration": {
        "town hall or registry office": ["a civil servant", "a resident"],
        "tax office enquiry": ["a tax officer", "a taxpayer"],
        "police report": ["a police officer", "a witness"],
        "council meeting statement": ["a councillor"],
    },
    "education": {
        "lecture or seminar": ["a university lecturer"],
        "parent-teacher meeting": ["a teacher", "a parent"],
        "student office hours": ["a professor", "a student"],
        "school announcement": ["a head teacher"],
    },
    "everyday conversation": {
        "phone call between friends": ["two friends"],
        "neighbours chatting": ["a neighbour"],
        "shopping and markets": ["a shopkeeper", "a shopper"],
        "making plans and appointments": ["a colleague", "a friend"],
    },
    "online video": {
        "YouTube tutorial or how-to": ["a YouTube creator"],
        "YouTube vlog or travel video": ["a vlogger"],
        "product review or unboxing": ["a reviewer"],
        "gaming or sports commentary": ["a commentator"],
        "explainer or documentary narration": ["a narrator"],
        "podcast or interview clip": ["a host", "a guest"],
        "news report": ["a news presenter", "a reporter"],
    },
    "film and series": {
        "family drama dialogue": ["a character in a drama series"],
        "comedy sketch or sitcom": ["a character in a comedy"],
        "crime thriller or police procedural": ["a detective", "a suspect"],
        "historical film": ["a character in a period film"],
        "romantic film dialogue": ["a character in a romantic film"],
        "science-fiction or fantasy dialogue": ["a character in a science-fiction film"],
    },
    "voice messages": {
        "family WhatsApp voice message": ["a person sending a voice message to a relative"],
        "friends' group chat voice message": ["a person sending a voice message to friends"],
        "Telegram voice message about work": ["a person sending a voice message to a colleague"],
        "voice message arranging a meeting or delivery": ["a person sending a voice message"],
        "voice message telling a story or news": ["a person sending a voice message"],
    },
}
REGISTERS = ["formal", "neutral", "colloquial", "hurried or stressed"]
# what each register sounds like in speech: the model otherwise writes polished prose for all of them
REGISTER_STYLE = {
    "formal": "polite, complete sentences, as a careful speaker would say them",
    "neutral": "plain conversational speech with some contractions",
    "colloquial": "relaxed everyday talk: contractions, fillers such as well, you know, like or I mean, "
    "trailing-off or restarted phrases, and informal wording",
    "hurried or stressed": "breathless, clipped talk: short fragments, interruptions, self-corrections and repeated words",
}

FIRST = [
    "Marianne",
    "Tomasz",
    "Ioana",
    "Giorgos",
    "Aoife",
    "Sven",
    "Elena",
    "Mateusz",
    "Katerina",
    "Lars",
    "Chiara",
    "Dimitri",
    "Petra",
    "Anselm",
    "Noor",
    "Beatrix",
    "Lucien",
    "Ingrid",
    "Rafael",
    "Zofia",
    "Gabriele",
    "Henrik",
    "Maja",
    "Cormac",
    "Stefania",
    "Joao",
    "Yasmin",
    "Viktor",
    "Elin",
    "Ahmed",
    "Paloma",
    "Oskar",
    "Leonie",
    "Matteo",
    "Dagmar",
    "Kasper",
    "Vera",
    "Niamh",
    "Radu",
    "Tuula",
    "Bruno",
    "Alma",
    "Istvan",
    "Kristina",
    "Emre",
    "Sofia",
    "Jan",
    "Mirela",
    "Teodor",
]
LAST = [
    "Kowalski",
    "Papadopoulos",
    "Murphy",
    "Lindqvist",
    "Rossi",
    "Novak",
    "Dubois",
    "Fernandez",
    "Schneider",
    "Horvat",
    "Jansen",
    "Silva",
    "Nagy",
    "Popescu",
    "Andersson",
    "Moreau",
    "Costa",
    "Virtanen",
    "Kaminski",
    "Ionescu",
    "Berg",
    "Vasquez",
    "Petrov",
    "Larsen",
    "Ricci",
    "Brandt",
    "Sokolov",
    "Doyle",
    "Hartmann",
    "Marinescu",
    "Eriksen",
    "Ferreira",
    "Szabo",
    "Lambert",
    "Kovacs",
    "Olsen",
    "Marchetti",
    "Weber",
    "Dimitrov",
]
CITIES = [
    "Lisbon",
    "Warsaw",
    "Vilnius",
    "Dublin",
    "Valletta",
    "Riga",
    "Tallinn",
    "Zagreb",
    "Ljubljana",
    "Bratislava",
    "Sofia",
    "Bucharest",
    "Athens",
    "Helsinki",
    "Copenhagen",
    "Stockholm",
    "Prague",
    "Budapest",
    "Kyiv",
    "Madrid",
    "Naples",
    "Lyon",
    "Antwerp",
    "Rotterdam",
    "Gdansk",
    "Porto",
    "Graz",
    "Cork",
    "Tampere",
    "Thessaloniki",
    "Plovdiv",
    "Kaunas",
    "Maribor",
    "Edinburgh",
    "Manchester",
    "Hamburg",
    "Milan",
    "Barcelona",
    "Seville",
    "Krakow",
    "Brno",
    "Oslo",
    "Bergen",
    "Reykjavik",
    "Vienna",
    "Salzburg",
    "Zurich",
    "Geneva",
    "Berlin",
    "Munich",
    "Cologne",
    "Amsterdam",
    "Brussels",
    "Ghent",
    "Paris",
    "Marseille",
    "Bordeaux",
    "Nice",
    "Rome",
    "Turin",
    "Florence",
    "Venice",
    "Palermo",
    "Bologna",
    "Valencia",
    "Bilbao",
    "Granada",
    "Faro",
    "Coimbra",
    "Gothenburg",
    "Malmo",
    "Aarhus",
    "Turku",
    "Oulu",
    "Tartu",
    "Daugavpils",
    "Klaipeda",
    "Poznan",
    "Wroclaw",
    "Lublin",
    "Ostrava",
    "Kosice",
    "Debrecen",
    "Cluj",
    "Timisoara",
    "Varna",
    "Split",
    "Dubrovnik",
    "Rijeka",
    "Sarajevo",
    "Belgrade",
    "Skopje",
    "Tirana",
    "Podgorica",
    "Chisinau",
    "Lviv",
    "Odesa",
    "Istanbul",
    "Nicosia",
    "Limassol",
    "Galway",
    "Belfast",
    "Glasgow",
    "Cardiff",
    "Leeds",
    "Bristol",
    "Lille",
    "Toulouse",
    "Strasbourg",
    "Luxembourg",
    "Gozo",
    "Mdina",
    "Palma",
    "Lausanne",
]
COUNTRIES = [
    "Portugal",
    "Spain",
    "France",
    "Italy",
    "Germany",
    "Austria",
    "Switzerland",
    "Belgium",
    "Netherlands",
    "Luxembourg",
    "Ireland",
    "Iceland",
    "Norway",
    "Sweden",
    "Finland",
    "Denmark",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Poland",
    "Czechia",
    "Slovakia",
    "Hungary",
    "Slovenia",
    "Croatia",
    "Romania",
    "Bulgaria",
    "Greece",
    "Cyprus",
    "Malta",
    "Serbia",
    "Albania",
    "Ukraine",
    "Moldova",
]
SITES = [
    "the Alhambra",
    "the Colosseum",
    "the Acropolis",
    "Sagrada Familia",
    "the Eiffel Tower",
    "Schonbrunn Palace",
    "the Charles Bridge",
    "Wawel Castle",
    "the Blue Lagoon",
    "Plitvice Lakes",
    "the Rila Monastery",
    "Bran Castle",
    "the Atomium",
    "the Tivoli Gardens",
    "Hallgrimskirkja",
    "the Vasa Museum",
    "Lake Balaton",
    "the Cliffs of Moher",
    "the Hal Saflieni Hypogeum",
    "the Hill of Crosses",
]
OPENINGS = [
    "a question",
    "an imperative or request",
    "a name or a greeting",
    "an interjection or filler such as well, so or oh",
    "a time or place expression",
    "a statement of fact",
    "a number or an amount",
    "a conjunction such as but, because or if",
]
MEDS = [
    "amoxicillin",
    "ibuprofen",
    "paracetamol",
    "metformin",
    "omeprazole",
    "salbutamol",
    "atorvastatin",
    "ramipril",
    "levothyroxine",
    "diclofenac",
    "loratadine",
    "amlodipine",
    "sertraline",
    "warfarin",
    "insulin",
    "glargine",
    "cetirizine",
    "naproxen",
    "clarithromycin",
    "lisinopril",
    "prednisolone",
]
THINGS = [
    "Eurostar",
    "Ryanair",
    "Lufthansa",
    "Vodafone",
    "Revolut",
    "IKEA",
    "Decathlon",
    "Zalando",
    "Booking.com",
    "Spotify",
    "Raiffeisen",
    "Santander",
    "Telekom",
    "Orange",
    "Iberia",
    "SNCF",
    "Trenitalia",
    "Deutsche",
    "Bahn",
    "Aer",
    "Lingus",
    "Air",
    "Baltic",
    "TAP",
    "Portugal",
    "Finnair",
    "KLM",
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def entity_hints(rng: random.Random, domain: str) -> str:
    people = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(2)]
    date = f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.choice([2024, 2025, 2026])}"
    extra = {"healthcare": f"{rng.randint(10, 500)} milligrams", "travel": f"platform {rng.randint(1, 12)}"}.get(
        domain, f"{rng.randint(2, 60)} minutes")  # fmt: skip
    amount = rng.choice([f"{rng.randint(5, 950)} euros", f"{rng.randint(2, 40)}.{rng.randint(10, 99)} euros", extra])
    wide = domain in ("travel", "online video")
    places = (
        f"{rng.choice(CITIES)}, {rng.choice(COUNTRIES)} or {rng.choice(SITES)}"
        if wide
        else f"{rng.choice(CITIES)} or {rng.choice(CITIES)}"
    )
    return (f"people such as {people[0]} or {people[1]}; places such as {places}; "
            f"the date {date}; a number or amount like {amount}; a name such as {rng.choice(MEDS + THINGS[:4] if domain == 'healthcare' else THINGS)}")  # fmt: skip


def build_prompts(n_prompts: int, per_prompt: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    combos = [(d, s, roles) for d, sc in DOMAINS.items() for s, roles in sc.items()]
    out = []
    for i in range(n_prompts):
        domain, scenario, roles = combos[i % len(combos)]
        register = REGISTERS[(i // len(combos)) % len(REGISTERS)]
        role = rng.choice(roles)
        content = (
            f"Write {per_prompt} different things that {role} might say aloud in this situation: {scenario}. "
            f"Register: {register} ({REGISTER_STYLE[register]}). Each item is one to three sentences of natural "
            f"spoken English, 12 to 40 words long, and different in wording and content from the others. Make them "
            f"sound spoken, not written, as in real {scenario} speech: transcribe how people actually talk, including "
            f"their disfluencies, never polished prose. The setting is Europe: amounts are in euros or other European "
            f"currencies, never dollars, and times use the 24-hour clock or o'clock. Make them concrete and rich "
            f"in specific names, places, dates, numbers, amounts and "
            f"{'medication or product' if domain == 'healthcare' else 'organisation or product'} names, for example "
            f"{entity_hints(rng, domain)}. Vary how the items begin: use {', '.join(rng.sample(OPENINGS, 3))}, "
            f"and never start more than two items with the same word. Output exactly one item per line, with no "
            f"numbering, no quotation marks and no commentary."
        )
        out.append({"prompt_id": i, "domain": domain, "scenario": scenario, "register": register,
                    "messages": [{"role": "user", "content": content}]})  # fmt: skip
    return out


def parse_lines(raw: str) -> list[str]:
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"“”').strip()
        if 10 <= len(line.split()) <= 60 and line[-1:] in ".?!":
            lines.append(line)
    return lines


def main(
    out: str,
    n_prompts: int = 40000,
    per_prompt: int = 10,
    seed: int = 0,
    temperature: float = 0.8,
    top_p: float = 0.95,
    gpu_mem: float = 0.88,
    ledger: str = "ledger.tsv",
    unit: str = "tts-text",
    model: str = MT_MODEL,
    tp: int = 1,
) -> None:
    from vllm import LLM, SamplingParams

    prompts = build_prompts(n_prompts, per_prompt, seed)
    t0 = time.time()
    llm = LLM(model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=gpu_mem, tensor_parallel_size=tp)
    ledger_append(ledger, unit=unit, stage="mt_load", n=0, gpu_seconds=round(time.time() - t0), gpus=1, status="ok")
    t1 = time.time()
    outs = llm.chat([p["messages"] for p in prompts],
                    SamplingParams(temperature=temperature, top_p=top_p, max_tokens=1200, seed=seed),
                    chat_template_kwargs={"enable_thinking": False})  # fmt: skip
    rows = []
    for p, o in zip(prompts, outs):
        for j, text in enumerate(parse_lines(o.outputs[0].text)):
            rows.append({"prompt_id": p["prompt_id"], "domain": p["domain"], "scenario": p["scenario"],
                         "register": p["register"], "text": text, "i": j})  # fmt: skip
    write_jsonl_atomic(Path(out), rows)
    ledger_append(
        ledger, unit=unit, stage="tts_text", n=len(rows), gpu_seconds=round(time.time() - t1), gpus=1, status="ok"
    )
    print(f"{len(prompts)} prompts -> {len(rows)} utterances")


if __name__ == "__main__":
    tyro.cli(main)
