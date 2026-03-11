"""
Natural Language Disambiguation Stimuli for Representation Experiments.

This module provides hand-crafted passages that are semantically ambiguous
between two interpretations (H1 and H2) until a single disambiguating word
forces one reading. Designed to test whether the velocity spike and preference
shift phenomena from synthetic graph walk experiments generalize to natural language.

Passage structure:
    [Ambiguous body] ~60-100 subword tokens consistent with EITHER interpretation
    [Disambiguating word] single word that forces one reading
    [Post-disambiguation] ~20-40 tokens clearly consistent with resolved interpretation

Usage:
    pairs = get_pilot_pairs()
    stimuli = get_all_stimuli()
    refs = get_reference_passages("cat_dog")
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class CategoryPair:
    """A binary category pair for disambiguation experiments."""
    name: str           # e.g., "cat_dog"
    h1_name: str        # e.g., "cat"
    h2_name: str        # e.g., "dog"
    shared_vocabulary: list[str]    # words valid for both interpretations
    h1_disambig_words: list[str]    # words unique to H1
    h2_disambig_words: list[str]    # words unique to H2


@dataclass
class DisambiguationStimulus:
    """A single disambiguation stimulus passage."""
    ambiguous_text: str         # The ambiguous portion
    disambig_word: str          # The single disambiguating word
    post_disambig_text: str     # Text after disambiguation
    true_hypothesis: str        # "H1" or "H2"
    category_pair: str          # e.g., "cat_dog"
    stimulus_id: str            # Unique identifier

    # Set after tokenization (default -1 = not yet tokenized)
    disambig_token_position: int = -1
    total_tokens: int = -1

    @property
    def full_text(self) -> str:
        """Complete passage text."""
        return f"{self.ambiguous_text} {self.disambig_word} {self.post_disambig_text}"

    @property
    def text_up_to_disambig(self) -> str:
        """Text before the disambiguating word (for no-disambig condition)."""
        return self.ambiguous_text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["full_text"] = self.full_text
        d["text_up_to_disambig"] = self.text_up_to_disambig
        return d


# ---------------------------------------------------------------------------
# Pilot category pairs
# ---------------------------------------------------------------------------

PILOT_PAIRS = [
    CategoryPair(
        name="cat_dog",
        h1_name="cat",
        h2_name="dog",
        shared_vocabulary=[
            "pet", "fur", "animal", "play", "sleep", "food", "owner",
            "cuddle", "soft", "warm", "paws", "tail", "eyes",
        ],
        h1_disambig_words=["purring", "meowed", "litter"],
        h2_disambig_words=["barking", "fetched", "leash"],
    ),
    CategoryPair(
        name="surgery_cooking",
        h1_name="surgery",
        h2_name="cooking",
        shared_vocabulary=[
            "preparation", "table", "precise", "instrument", "careful",
            "heat", "cut", "hands", "clean", "technique", "assistant",
        ],
        h1_disambig_words=["scalpel", "sutures", "anesthesia"],
        h2_disambig_words=["spatula", "seasoning", "simmered"],
    ),
    CategoryPair(
        name="chess_war",
        h1_name="chess",
        h2_name="war",
        shared_vocabulary=[
            "strategy", "position", "attack", "defense", "sacrifice",
            "opponent", "advance", "retreat", "victory", "flank",
        ],
        h1_disambig_words=["checkmate", "castled", "pawn"],
        h2_disambig_words=["artillery", "battalion", "trenches"],
    ),
    CategoryPair(
        name="painting_photography",
        h1_name="painting",
        h2_name="photography",
        shared_vocabulary=[
            "composition", "light", "subject", "frame", "gallery",
            "capture", "detail", "perspective", "color", "studio",
        ],
        h1_disambig_words=["brushstroke", "canvas", "palette"],
        h2_disambig_words=["shutter", "lens", "aperture"],
    ),
]


# ---------------------------------------------------------------------------
# Hand-crafted stimuli: 6 per pair (3 H1, 3 H2)
# ---------------------------------------------------------------------------

_STIMULI = {
    "cat_dog": [
        # --- H1 (cat) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "The animal stretched out on the couch, its soft fur brushing "
                "against my arm. It loved being near people, always seeking "
                "attention and warmth. When I moved to the kitchen, it followed "
                "close behind, watching intently as I opened a can of food. "
                "It made soft sounds, nudging my hand with its head, then "
                "curled up beside me and started"
            ),
            disambig_word="purring",
            post_disambig_text=(
                "softly, the gentle vibration spreading through its whole body. "
                "Its whiskers twitched as it kneaded the blanket beneath its "
                "paws, settling deeper into the cushion."
            ),
            true_hypothesis="H1",
            category_pair="cat_dog",
            stimulus_id="cat_dog_h1_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "Every morning the little creature would wait by the door, "
                "ears perked and tail swaying gently. It had been part of the "
                "family for years now, a constant companion through good times "
                "and bad. I reached down to stroke its warm fur, feeling it "
                "press against my palm. It looked up at me with those bright "
                "eyes and suddenly"
            ),
            disambig_word="meowed",
            post_disambig_text=(
                "in a high-pitched tone, demanding breakfast. It weaved between "
                "my legs as I walked to the cupboard, its tail held high like "
                "a question mark."
            ),
            true_hypothesis="H1",
            category_pair="cat_dog",
            stimulus_id="cat_dog_h1_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "The new pet adjusted quickly to its surroundings, exploring "
                "every corner of the apartment with quiet curiosity. It claimed "
                "the sunny spot by the window as its favorite place, spending "
                "hours watching the world outside. My roommate worried about the "
                "smell, so I made sure to clean the"
            ),
            disambig_word="litter",
            post_disambig_text=(
                "box every day, scooping out the clumps and adding fresh clay. "
                "The pet seemed to appreciate the cleanliness, always grooming "
                "itself meticulously after each visit."
            ),
            true_hypothesis="H1",
            category_pair="cat_dog",
            stimulus_id="cat_dog_h1_2",
        ),
        # --- H2 (dog) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "The animal bounded through the park, full of energy and "
                "excitement. Its fur gleamed in the afternoon sun as it ran "
                "in circles around the other visitors. I called out and it "
                "came racing back, tail wagging wildly. I threw a stick across "
                "the grass and it immediately"
            ),
            disambig_word="fetched",
            post_disambig_text=(
                "it, sprinting back with the stick clamped firmly between its "
                "jaws, dropping it at my feet and crouching down eagerly for "
                "another throw."
            ),
            true_hypothesis="H2",
            category_pair="cat_dog",
            stimulus_id="cat_dog_h2_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "Our pet had always been protective of the family, alert to "
                "every sound and movement near the house. At night it would "
                "patrol the hallway, checking on each room before settling "
                "down. When the mail carrier arrived that morning, it leapt "
                "to the window and started"
            ),
            disambig_word="barking",
            post_disambig_text=(
                "loudly, its deep voice echoing through the house. I grabbed "
                "the collar and clipped on the leash to take it for a walk "
                "around the block."
            ),
            true_hypothesis="H2",
            category_pair="cat_dog",
            stimulus_id="cat_dog_h2_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "Taking the new pet outside was part of our daily routine. "
                "It needed fresh air and exercise, otherwise it would get "
                "restless and start chewing on furniture. I grabbed my jacket "
                "and reached for the"
            ),
            disambig_word="leash",
            post_disambig_text=(
                "hanging by the door. The moment it heard the clip jingle, "
                "it bounded over, spinning in circles of excitement, ready "
                "for its afternoon walk through the neighborhood."
            ),
            true_hypothesis="H2",
            category_pair="cat_dog",
            stimulus_id="cat_dog_h2_2",
        ),
    ],
    "surgery_cooking": [
        # --- H1 (surgery) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "The room was sterile and brightly lit, every surface wiped "
                "clean. She put on gloves and laid out the instruments on the "
                "metal table, checking each one carefully. Her assistant stood "
                "ready, handing her whatever she needed. She examined the "
                "subject closely, then made the first careful cut with the"
            ),
            disambig_word="scalpel",
            post_disambig_text=(
                "tracing a precise line along the marked incision site. Blood "
                "welled up immediately, and her assistant applied suction while "
                "she widened the opening to expose the tissue beneath."
            ),
            true_hypothesis="H1",
            category_pair="surgery_cooking",
            stimulus_id="surgery_cooking_h1_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "Preparation was everything. She reviewed the plan one more "
                "time, noting every detail that would matter in the next few "
                "hours. The team assembled around the table, each person "
                "knowing their role. Temperature had to be controlled precisely "
                "throughout the process. She picked up her instruments and "
                "began, knowing there would be no room for error once the"
            ),
            disambig_word="anesthesia",
            post_disambig_text=(
                "took full effect. The patient's vitals stabilized on the "
                "monitor as she made her approach, years of training guiding "
                "her steady hands through the delicate procedure."
            ),
            true_hypothesis="H1",
            category_pair="surgery_cooking",
            stimulus_id="surgery_cooking_h1_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "He worked methodically, his hands moving with practiced "
                "precision. Years of training had made these movements second "
                "nature. The assistant passed him each tool as needed, "
                "anticipating the next step. He paused to inspect his work, "
                "making sure everything was aligned perfectly before closing "
                "things up with neat"
            ),
            disambig_word="sutures",
            post_disambig_text=(
                "along the wound edge, each stitch placed exactly three "
                "millimeters apart. The patient would heal cleanly, with "
                "minimal scarring if all went according to plan."
            ),
            true_hypothesis="H1",
            category_pair="surgery_cooking",
            stimulus_id="surgery_cooking_h1_2",
        ),
        # --- H2 (cooking) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "The room was hot and brightly lit, every surface wiped "
                "clean. She put on gloves and laid out her tools on the metal "
                "table, checking each one carefully. Her assistant stood ready, "
                "handing her what she needed. She examined the ingredients "
                "closely, then reached for the"
            ),
            disambig_word="spatula",
            post_disambig_text=(
                "and flipped the golden crepe in one smooth motion. The "
                "butter sizzled as it hit the hot surface, filling the "
                "kitchen with a rich, nutty aroma."
            ),
            true_hypothesis="H2",
            category_pair="surgery_cooking",
            stimulus_id="surgery_cooking_h2_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "Preparation was everything. He reviewed the plan one more "
                "time, checking the quantities and noting every step. His "
                "team gathered around the long table, each person knowing "
                "their role. Timing would be critical — too much heat or too "
                "little would ruin the result. He picked up his tools and "
                "began adding the"
            ),
            disambig_word="seasoning",
            post_disambig_text=(
                "blend of cumin, paprika, and black pepper, tossing it over "
                "the meat. The fragrant spices bloomed in the hot oil, "
                "transforming the simple dish into something extraordinary."
            ),
            true_hypothesis="H2",
            category_pair="surgery_cooking",
            stimulus_id="surgery_cooking_h2_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "She worked methodically, her hands moving with practiced "
                "precision. Everything had to be done in the right order. "
                "The heat was intense, and she adjusted the temperature with "
                "a careful turn. She added the liquid slowly, watching as "
                "the mixture transformed. Then she lowered the flame and "
                "let everything"
            ),
            disambig_word="simmered",
            post_disambig_text=(
                "gently for twenty minutes, the sauce thickening and reducing "
                "to a glossy consistency. She tasted it, added a pinch of "
                "salt, and smiled — dinner was nearly ready."
            ),
            true_hypothesis="H2",
            category_pair="surgery_cooking",
            stimulus_id="surgery_cooking_h2_2",
        ),
    ],
    "chess_war": [
        # --- H1 (chess) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "The confrontation had been building for hours. Both sides "
                "maneuvered carefully, probing for weaknesses in the opponent's "
                "position. A bold sacrifice opened up the flank, but the "
                "defense held firm. The tension in the room was palpable. "
                "Then, in a brilliant move that no one had anticipated, he "
                "delivered"
            ),
            disambig_word="checkmate",
            post_disambig_text=(
                "with his knight, forking the king and the last remaining "
                "rook. His opponent stared at the board in disbelief, then "
                "slowly tipped over his king in resignation."
            ),
            true_hypothesis="H1",
            category_pair="chess_war",
            stimulus_id="chess_war_h1_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "The opening moves were cautious, both sides developing their "
                "forces methodically. Control of the center was the immediate "
                "priority, with neither side willing to overextend. As the "
                "struggle intensified, he saw an opportunity to protect his "
                "king by shifting his defenses. He"
            ),
            disambig_word="castled",
            post_disambig_text=(
                "kingside, tucking his king safely behind a wall of pawns "
                "while activating the rook on the open file. The positional "
                "advantage was now clearly in his favor."
            ),
            true_hypothesis="H1",
            category_pair="chess_war",
            stimulus_id="chess_war_h1_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "She studied the position carefully, weighing her options. "
                "The opponent had committed forces to the left side, leaving "
                "the center vulnerable. A direct assault could break through, "
                "but it required sacrificing something first. She pushed her "
                "most expendable unit forward, advancing the"
            ),
            disambig_word="pawn",
            post_disambig_text=(
                "two squares to e4, seizing control of the center. The gambit "
                "was accepted, but now her bishops had clear diagonals and "
                "the queenside attack was unstoppable."
            ),
            true_hypothesis="H1",
            category_pair="chess_war",
            stimulus_id="chess_war_h1_2",
        ),
        # --- H2 (war) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "The confrontation had been building for months. Both sides "
                "maneuvered carefully, probing for weaknesses in the opponent's "
                "defenses. A bold offensive opened up the eastern flank, but "
                "the lines held firm. The tension in the command center was "
                "palpable. Then the general ordered the"
            ),
            disambig_word="artillery",
            post_disambig_text=(
                "to commence firing, and the horizon erupted in plumes of "
                "smoke and flame. The bombardment shook the ground for miles, "
                "shattering the fortified positions along the ridge."
            ),
            true_hypothesis="H2",
            category_pair="chess_war",
            stimulus_id="chess_war_h2_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "The advance had stalled, and the commander reassessed the "
                "situation. The opponent controlled the high ground, and "
                "any frontal assault would be costly. He needed to outflank "
                "them. He ordered his reserves forward, sending the entire"
            ),
            disambig_word="battalion",
            post_disambig_text=(
                "through the mountain pass under cover of darkness. By dawn, "
                "twelve hundred soldiers had encircled the enemy position, "
                "cutting off their supply lines completely."
            ),
            true_hypothesis="H2",
            category_pair="chess_war",
            stimulus_id="chess_war_h2_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "The enemy launched a devastating attack on the left flank, "
                "overwhelming the forward positions. Retreat was the only "
                "option if they wanted to survive. The commander ordered his "
                "forces to fall back and dig into the"
            ),
            disambig_word="trenches",
            post_disambig_text=(
                "that had been prepared along the river bank. Soldiers piled "
                "sandbags and strung barbed wire through the night, preparing "
                "for the counterattack that would come at dawn."
            ),
            true_hypothesis="H2",
            category_pair="chess_war",
            stimulus_id="chess_war_h2_2",
        ),
    ],
    "painting_photography": [
        # --- H1 (painting) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "She spent the morning setting up in the studio, arranging "
                "the subject under the best possible light. The composition "
                "needed to be perfect — every detail, every shadow had to "
                "contribute to the whole. She adjusted the angle slightly, "
                "stepped back to assess the framing, and then began to work, "
                "applying the first"
            ),
            disambig_word="brushstroke",
            post_disambig_text=(
                "with a wide, confident sweep of cerulean blue across the "
                "upper third. The oil paint caught the studio light, its "
                "texture building the suggestion of a summer sky."
            ),
            true_hypothesis="H1",
            category_pair="painting_photography",
            stimulus_id="painting_photography_h1_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "The gallery opening was tomorrow, and he still had one piece "
                "to finish. He had been working on it for weeks, layering "
                "color and texture to capture the exact quality of light he "
                "remembered from that afternoon in Provence. He picked up "
                "his materials and returned to the"
            ),
            disambig_word="canvas",
            post_disambig_text=(
                "propped against the easel, mixing a warm ochre into the "
                "existing underpainting. The linen surface drank in the "
                "pigment as he built up the golden light of the hillside."
            ),
            true_hypothesis="H1",
            category_pair="painting_photography",
            stimulus_id="painting_photography_h1_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "Getting the colors right was the hardest part. The subject "
                "looked different depending on the time of day, and she "
                "needed to capture a specific mood. She mixed and tested, "
                "adjusted and compared, until the tones matched her vision "
                "exactly. Satisfied, she loaded her tools from the"
            ),
            disambig_word="palette",
            post_disambig_text=(
                "and began applying thin glazes of viridian green, letting "
                "each translucent layer dry before adding the next. The "
                "depth of color that emerged was impossible to achieve any "
                "other way."
            ),
            true_hypothesis="H1",
            category_pair="painting_photography",
            stimulus_id="painting_photography_h1_2",
        ),
        # --- H2 (photography) stimuli ---
        DisambiguationStimulus(
            ambiguous_text=(
                "She spent the morning setting up in the studio, arranging "
                "the subject under the best possible light. The composition "
                "needed to be perfect — every detail, every shadow had to "
                "contribute to the whole. She adjusted the angle slightly, "
                "stepped back to assess the framing, and then pressed the"
            ),
            disambig_word="shutter",
            post_disambig_text=(
                "button, capturing the image in a fraction of a second. She "
                "reviewed the shot on the display — the exposure was perfect, "
                "every highlight and shadow exactly where she wanted them."
            ),
            true_hypothesis="H2",
            category_pair="painting_photography",
            stimulus_id="painting_photography_h2_0",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "The gallery opening was tomorrow, and he still had work to "
                "do. He reviewed his recent images on the large monitor, "
                "adjusting contrast and saturation until each one matched "
                "his vision. He swapped to a wider"
            ),
            disambig_word="lens",
            post_disambig_text=(
                "for the final series, wanting to capture more of the "
                "environment around his subjects. The shallow depth of field "
                "would separate them beautifully from the urban backdrop."
            ),
            true_hypothesis="H2",
            category_pair="painting_photography",
            stimulus_id="painting_photography_h2_1",
        ),
        DisambiguationStimulus(
            ambiguous_text=(
                "Getting the light right was the hardest part. The subject "
                "looked different depending on the time of day, and she "
                "needed to capture a specific mood. She adjusted the settings "
                "carefully, narrowing the"
            ),
            disambig_word="aperture",
            post_disambig_text=(
                "to f/8 for maximum sharpness across the frame. The slower "
                "shutter speed required a tripod, but the crisp detail in "
                "the resulting image was worth the extra setup time."
            ),
            true_hypothesis="H2",
            category_pair="painting_photography",
            stimulus_id="painting_photography_h2_2",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Reference passages (5 per hypothesis per pair)
# ---------------------------------------------------------------------------

_REFERENCE_PASSAGES = {
    "cat_dog": {
        "H1": [
            "The tabby cat groomed itself on the windowsill, licking its paws clean and purring contentedly in the warm afternoon sunlight.",
            "She adopted a kitten from the shelter, a tiny calico that loved to chase feather toys and nap in cardboard boxes.",
            "The cat stretched out on the radiator, its tail curling lazily as it watched birds through the window with half-closed eyes.",
            "Every evening the old tomcat would sit on the porch, meowing softly until someone let him inside for dinner and a warm lap.",
            "The Siamese cat knocked a glass off the counter, then sat atop the refrigerator looking supremely unbothered by the mess below.",
        ],
        "H2": [
            "The golden retriever bounded across the yard, fetching the tennis ball and dropping it at my feet, tail wagging furiously.",
            "She took the puppy to obedience class, where it learned to sit, stay, and walk on a leash without pulling ahead.",
            "The old hound dozed on the porch, occasionally lifting its head to bark at passing squirrels before settling back down.",
            "Every morning the border collie waited by the door, leash in its mouth, ready for the long walk through the neighborhood.",
            "The German shepherd stood guard at the gate, its ears pricked forward, alert to every sound in the quiet street.",
        ],
    },
    "surgery_cooking": {
        "H1": [
            "The surgeon scrubbed in and reviewed the patient's imaging one final time before making the initial incision along the marked line.",
            "The operation lasted four hours, during which the team carefully removed the tumor while monitoring the patient's vital signs continuously.",
            "He sutured the wound closed with precise stitches, then applied sterile dressings and briefed the family on the recovery timeline.",
            "The anesthesiologist adjusted the dosage as the procedure entered its most delicate phase, keeping the patient safely unconscious throughout.",
            "After the surgery, the patient was wheeled to the recovery room where nurses monitored blood pressure, heart rate, and oxygen levels.",
        ],
        "H2": [
            "The chef diced the onions finely and tossed them into the hot pan, where they sizzled and turned golden within minutes.",
            "She followed the recipe carefully, measuring out flour, sugar, and butter before folding everything together into a smooth batter.",
            "The pasta water came to a rolling boil and he lowered in the fresh noodles, stirring gently to keep them from sticking.",
            "He plated the dish with care, drizzling balsamic reduction around the edges and finishing with a sprig of fresh rosemary.",
            "The kitchen filled with the aroma of roasting garlic and herbs as the lamb braised slowly in the cast iron pot.",
        ],
    },
    "chess_war": {
        "H1": [
            "The grandmaster opened with the Sicilian Defense, moving his pawn to c5 in response to white's king pawn opening at e4.",
            "She sacrificed her bishop to break open the kingside, creating a devastating attack that led to checkmate in five moves.",
            "The endgame was a classic rook and pawn position, with both players maneuvering carefully to promote their passed pawns first.",
            "He studied the board for ten minutes before finding the winning combination, a quiet knight move that trapped the opposing queen.",
            "The tournament hall was silent as the two finalists played the last game, their chess clocks ticking down to the final seconds.",
        ],
        "H2": [
            "The infantry regiment advanced under covering fire from the tanks, pushing through the destroyed village toward the strategic crossroads.",
            "Artillery shells rained down on the fortified positions for hours before the ground assault began at dawn the following morning.",
            "The general deployed three battalions along the river, establishing defensive positions with trenches, bunkers, and overlapping fields of fire.",
            "Air support was called in when the enemy counterattack threatened to overrun the forward operating base on the eastern ridge.",
            "Casualties were heavy in the initial assault, but the soldiers held their ground until reinforcements arrived by helicopter that evening.",
        ],
    },
    "painting_photography": {
        "H1": [
            "She mixed cadmium yellow with titanium white on her palette, then applied thick impasto strokes to build up the texture of sunflowers.",
            "The oil painting took three months to complete, with each layer of glaze adding luminosity and depth to the portrait.",
            "He cleaned his brushes with turpentine and stepped back from the easel to study the landscape he had been working on all morning.",
            "The watercolor bled beautifully across the damp paper, creating soft gradients of blue and purple that suggested distant mountains.",
            "She primed the canvas with gesso and sketched the composition in charcoal before beginning to block in the major color areas.",
        ],
        "H2": [
            "He adjusted the shutter speed to one-sixtieth of a second and opened the aperture wide to blur the background behind the portrait.",
            "The DSLR camera captured every detail of the landscape, from the sharp foreground rocks to the soft clouds on the horizon.",
            "She reviewed the photographs on her laptop, selecting the best exposures and adjusting the white balance in post-processing software.",
            "The long exposure turned the waterfall into a silky ribbon of white, while the surrounding rocks remained perfectly sharp in the image.",
            "He switched to a telephoto lens to capture the bird from a distance, keeping the autofocus locked on its eye as it perched.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_pilot_pairs() -> list[CategoryPair]:
    """Return the 4 pilot category pairs."""
    return PILOT_PAIRS.copy()


def get_pair_by_name(name: str) -> CategoryPair:
    """Get a specific category pair by name."""
    for pair in PILOT_PAIRS:
        if pair.name == name:
            return pair
    raise ValueError(f"Unknown pair: {name}. Available: {[p.name for p in PILOT_PAIRS]}")


def get_stimuli(pair_name: str) -> list[DisambiguationStimulus]:
    """Get all stimuli for a given category pair."""
    if pair_name not in _STIMULI:
        raise ValueError(f"No stimuli for pair: {pair_name}")
    return _STIMULI[pair_name].copy()


def get_all_stimuli() -> list[DisambiguationStimulus]:
    """Get all stimuli across all pairs."""
    all_stim = []
    for pair in PILOT_PAIRS:
        all_stim.extend(get_stimuli(pair.name))
    return all_stim


def get_reference_passages(pair_name: str) -> dict[str, list[str]]:
    """Get reference passages for a category pair. Returns {"H1": [...], "H2": [...]}."""
    if pair_name not in _REFERENCE_PASSAGES:
        raise ValueError(f"No reference passages for pair: {pair_name}")
    return _REFERENCE_PASSAGES[pair_name].copy()


def tokenize_stimulus(stimulus: DisambiguationStimulus, tokenizer) -> DisambiguationStimulus:
    """
    Tokenize a stimulus and fill in token position fields.

    Finds the exact subword token position where the disambiguating word begins.
    """
    # Tokenize text up to disambig word
    pre_tokens = tokenizer.encode(stimulus.ambiguous_text, add_special_tokens=False)
    # Tokenize full text
    full_tokens = tokenizer.encode(stimulus.full_text, add_special_tokens=False)

    stimulus.disambig_token_position = len(pre_tokens)
    stimulus.total_tokens = len(full_tokens)
    return stimulus


def validate_stimulus(
    stimulus: DisambiguationStimulus,
    model,
    tokenizer,
    top_k: int = 20,
) -> tuple[bool, int, list[str]]:
    """
    Check if the disambiguating word leaks through the ambiguous context.

    Returns:
        (is_valid, disambig_word_rank, top_k_words):
        - is_valid: True if disambig word is NOT in the model's top-k predictions
        - disambig_word_rank: Rank of the disambig word (0 = most likely), -1 if not found
        - top_k_words: The actual top-k predicted words
    """
    import torch

    # Tokenize the ambiguous text (context before disambig word)
    input_ids = tokenizer.encode(stimulus.ambiguous_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    # Get predictions at the last position
    with torch.no_grad():
        logits = model.model(input_ids).logits[0, -1, :]

    # Get top-k token IDs
    top_k_ids = torch.topk(logits, k=min(top_k * 5, logits.shape[0])).indices.tolist()
    top_k_tokens = [tokenizer.decode([tid]).strip().lower() for tid in top_k_ids]

    # Check if disambig word appears in top-k
    disambig_lower = stimulus.disambig_word.lower()
    disambig_rank = -1
    for i, tok in enumerate(top_k_tokens):
        if disambig_lower in tok or tok in disambig_lower:
            disambig_rank = i
            break

    is_valid = disambig_rank < 0 or disambig_rank >= top_k
    top_k_words = top_k_tokens[:top_k]

    return is_valid, disambig_rank, top_k_words


def build_instruct_prompt(
    stimulus: DisambiguationStimulus,
    pair: CategoryPair,
    tokenizer,
    include_disambig: bool = True,
) -> str:
    """
    Build an instruct-model prompt with question framing.

    Returns the formatted prompt string using the tokenizer's chat template.
    """
    question = f"Is the following passage about {pair.h1_name} or {pair.h2_name}?"

    if include_disambig:
        passage = stimulus.full_text
    else:
        passage = stimulus.ambiguous_text

    messages = [
        {"role": "user", "content": f"{question}\n\n{passage}"},
    ]

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback for base models without chat template
        return f"Q: {question}\n\n{passage}\nA:"


def save_stimuli_to_json(stimuli: list[DisambiguationStimulus], path: str | Path):
    """Save stimuli to a JSON file for reproducibility."""
    data = [s.to_dict() for s in stimuli]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_stimuli_from_json(path: str | Path) -> list[DisambiguationStimulus]:
    """Load stimuli from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    stimuli = []
    for d in data:
        s = DisambiguationStimulus(
            ambiguous_text=d["ambiguous_text"],
            disambig_word=d["disambig_word"],
            post_disambig_text=d["post_disambig_text"],
            true_hypothesis=d["true_hypothesis"],
            category_pair=d["category_pair"],
            stimulus_id=d["stimulus_id"],
            disambig_token_position=d.get("disambig_token_position", -1),
            total_tokens=d.get("total_tokens", -1),
        )
        stimuli.append(s)
    return stimuli
