# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world recommenders (Spotify, YouTube, Netflix) mostly learn from *behavior* — what
millions of people play, skip, save, and replay — and combine that with **content signals**
about the items themselves and **collaborative signals** about which users resemble each
other. They optimize for engagement over huge catalogs using machine-learned models that no
person could write by hand. My version is a deliberately small, transparent slice of that
idea: a **content-based** recommender that ignores popularity and other users entirely and
scores each song purely on **how closely its attributes match a single user's stated taste**.
It prioritizes *explainability and vibe-matching* over scale or personalization-from-history —
every recommendation can point to the exact features that earned it, so I can reason about
*why* a song was chosen rather than trusting a black box.

### Features used

**`Song`** stores each track's attributes (from `data/songs.csv`):

| Field | Type | Role in scoring |
|---|---|---|
| `id`, `title`, `artist` | identifiers | display only, not scored |
| `genre` | category | soft match (low weight) |
| `mood` | category | vibe match (higher weight than genre) |
| `energy` | float 0–1 | closeness to user's `target_energy` |
| `valence` | float 0–1 | emotional positivity |
| `danceability` | float 0–1 | groove |
| `acousticness` | float 0–1 | organic vs. electronic texture |
| `tempo_bpm` | int (60–152) | pace — normalized before scoring |

**`UserProfile`** stores the user's taste preferences:

- `favorite_genre` — the genre to reward on a match
- `favorite_mood` — the mood/vibe to reward on a match
- `target_energy` — the *ideal* energy level (songs are scored on closeness to it, not on being higher)
- `likes_acoustic` — whether to favor acoustic tracks

### Scoring Logic (the Algorithm Recipe)

For each song the `Recommender` applies a **Scoring Rule** that awards points, then a
**Ranking Rule** that sorts and trims. Categorical features are all-or-nothing; numeric
features are graded by closeness (`1 − distance`) so songs *near* the target still earn most
of their points.

| Rule | Points | Formula |
|---|---|---|
| Genre match | +2.0 | `+2.0 if song.genre == favorite_genre else 0` |
| Mood match | +1.0 | `+1.0 if song.mood == favorite_mood else 0` |
| Energy closeness | 0 → +2.0 | `2.0 * (1 - abs(song.energy - target_energy))` |
| Acoustic preference | 0 → +1.0 | if `likes_acoustic`: `1.0 * acousticness`, else `1.0 * (1 - acousticness)` |

**Maximum score = 6.0.** This recipe is deliberately **genre-heavy** — a genre match is worth
double a mood match — so recommendations stay stylistically consistent with the user's
favorite genre. (Flipping genre and mood weights is the first planned experiment below.)

**Ranking Rule:** score all songs, sort by score descending, return the top *k* (default 5),
breaking ties by energy closeness.

### Potential Biases I Expect

- **Genre over-prioritization.** Because a genre match (+2.0) outweighs a mood match (+1.0),
  the system may over-prioritize genre and ignore great songs that match the user's *mood* but
  sit in a different genre — e.g. a "chill" jazz track passed over for a lofi track just
  because the user's favorite genre is lofi.
- **Popularity/diversity blind spot.** Scoring is purely content-based, so a user locked to one
  genre + mood will get a narrow, self-reinforcing list. There's no mechanism to surface
  variety or discovery.
- **Catalog-composition bias.** With only 18 songs, whichever genre has the most entries has a
  built-in advantage; an under-represented genre can't win even for a user who'd love it.
- **Feature-weighting is a value judgment.** The chosen weights encode *my* assumption about
  what matters. A different weighting would produce different "best" songs — the recipe is a
  hypothesis, not ground truth.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Produced by `python -m src.main` using the "chill lofi" profile
(`favorite_genre=lofi, favorite_mood=chill, target_energy=0.35, likes_acoustic=True`):

```
Loaded songs: 18

User profile:
  favorite_genre: lofi
  favorite_mood: chill
  target_energy: 0.35
  likes_acoustic: True

====================================================
Top recommendations
====================================================

1. Library Rain - Paper Lanterns
   Genre: lofi | Mood: chill
   Score: 5.86 / 6.00
   Reasons:
     - genre match: lofi (+2.0)
     - mood match: chill (+1.0)
     - energy 0.35 vs target 0.35 (+2.00)
     - acoustic, user likes acoustic (+0.86)

2. Midnight Coding - LoRoom
   Genre: lofi | Mood: chill
   Score: 5.57 / 6.00
   Reasons:
     - genre match: lofi (+2.0)
     - mood match: chill (+1.0)
     - energy 0.42 vs target 0.35 (+1.86)
     - acoustic, user likes acoustic (+0.71)

3. Focus Flow - LoRoom
   Genre: lofi | Mood: focused
   Score: 4.68 / 6.00
   Reasons:
     - genre match: lofi (+2.0)
     - energy 0.40 vs target 0.35 (+1.90)
     - acoustic, user likes acoustic (+0.78)

4. Spacewalk Thoughts - Orbit Bloom
   Genre: ambient | Mood: chill
   Score: 3.78 / 6.00
   Reasons:
     - mood match: chill (+1.0)
     - energy 0.28 vs target 0.35 (+1.86)
     - acoustic, user likes acoustic (+0.92)

5. Paper Boats - Willow Hart
   Genre: folk | Mood: dreamy
   Score: 2.88 / 6.00
   Reasons:
     - energy 0.34 vs target 0.35 (+1.98)
     - acoustic, user likes acoustic (+0.90)
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Stress Test: Diverse User Profiles

To evaluate the scoring logic, `src/main.py` runs five profiles: three normal
tastes and two adversarial / edge-case profiles built to try to "trick" the
scorer. Full output of `python -m src.main` below.

### 1. High-Energy Pop  (pop / happy / energy 0.9 / non-acoustic)

```
1. Sunrise City - Neon Echo        Score: 5.66  (genre+2, mood+1, energy+1.84, non-acoustic+0.82)
2. Gym Hero - Max Pulse            Score: 4.89  (genre+2, energy+1.94, non-acoustic+0.95)
3. Rooftop Lights - Indigo Parade  Score: 3.37  (mood+1, energy+1.72, non-acoustic+0.65)
4. Storm Runner - Voltline         Score: 2.88  (energy+1.98, non-acoustic+0.90)
5. Pulse Machine - Hexadrive       Score: 2.87  (energy+1.90, non-acoustic+0.97)
```
Works as intended: the two pop tracks dominate; high-energy non-pop songs fill the tail.

### 2. Chill Lofi  (lofi / chill / energy 0.35 / acoustic)

```
1. Library Rain - Paper Lanterns   Score: 5.86  (genre+2, mood+1, energy+2.00, acoustic+0.86)
2. Midnight Coding - LoRoom        Score: 5.57  (genre+2, mood+1, energy+1.86, acoustic+0.71)
3. Focus Flow - LoRoom             Score: 4.68  (genre+2, energy+1.90, acoustic+0.78)
4. Spacewalk Thoughts - Orbit Bloom Score: 3.78 (mood+1, energy+1.86, acoustic+0.92)
5. Paper Boats - Willow Hart       Score: 2.88  (energy+1.98, acoustic+0.90)
```
Library Rain scores a perfect energy match (0.35 = 0.35). All lofi tracks rank top.

### 3. Deep Intense Rock  (rock / intense / energy 0.9 / non-acoustic)

```
1. Storm Runner - Voltline         Score: 5.88  (genre+2, mood+1, energy+1.98, non-acoustic+0.90)
2. Gym Hero - Max Pulse            Score: 3.89  (mood+1, energy+1.94, non-acoustic+0.95)
3. Pulse Machine - Hexadrive       Score: 2.87  (energy+1.90, non-acoustic+0.97)
4. Iron Verdict - Blacklung Rite   Score: 2.82  (energy+1.84, non-acoustic+0.98)
5. Concrete Kingdom - Vault Kid    Score: 2.82  (energy+1.90, non-acoustic+0.92)
```
Only one rock song exists, so the tail fills with other high-energy, non-acoustic genres.

### 4. ADVERSARIAL — Conflicted  (classical / melancholy / energy 0.9 / acoustic)

Conflicting preferences: wants **high energy (0.9)** but a **melancholy** mood, which in this
catalog only exists on a *low-energy* classical track.

```
1. Winter Elegy - Aria Sever       Score: 4.75  (genre+2, mood+1, energy+0.80, acoustic+0.95)
2. Storm Runner - Voltline         Score: 2.08  (energy+1.98, acoustic+0.10)
3. Rooftop Lights - Indigo Parade  Score: 2.07  (energy+1.72, acoustic+0.35)
4. Sunrise City - Neon Echo        Score: 2.02  (energy+1.84, acoustic+0.18)
5. Gym Hero - Max Pulse            Score: 1.99  (energy+1.94, acoustic+0.05)
```
**Finding:** the categorical +3 (genre+mood) overpowers the energy mismatch — Winter Elegy wins
*despite* being the opposite energy the user asked for. The scorer resolves the conflict in favor
of genre/mood, exposing the genre-heavy bias documented above.

### 5. EDGE CASE — Impossible  (polka / sad / energy 0.5 / non-acoustic)

A genre and mood that **don't exist** in the catalog, so no categorical points can ever fire.

```
1. Velvet Hours - Mona Skye        Score: 2.50  (energy+1.90, non-acoustic+0.60)
2. Island Time - Sunny Roots       Score: 2.45  (energy+1.80, non-acoustic+0.65)
3. Night Drive Loop - Neon Echo    Score: 2.28  (energy+1.50, non-acoustic+0.78)
4. Concrete Kingdom - Vault Kid    Score: 2.22  (energy+1.30, non-acoustic+0.92)
5. Dust Road Memory - Colt Harlow  Score: 2.22  (energy+1.90, non-acoustic+0.32)
```
**Finding:** it degrades gracefully — no crash, no empty list. Ranking falls back entirely to
the numeric features (energy + acoustic), so the user still gets sensible mid-energy songs even
though nothing matched their stated genre/mood.

---

## Experiments You Tried

### Weight Shift: double `energy`, halve `genre`

I refactored the weights into named constants (`GENRE_WEIGHT`, `MOOD_WEIGHT`, `ENERGY_WEIGHT`,
`ACOUSTIC_WEIGHT` in `src/recommender.py`) and re-ran all five profiles with
`GENRE_WEIGHT = 1.0` (down from 2.0) and `ENERGY_WEIGHT = 4.0` (up from 2.0).

Baseline top-1 vs. experiment top-1:

| Profile | Baseline #1 | Experiment #1 | Changed? |
|---|---|---|---|
| High-Energy Pop | Sunrise City (5.66) | Sunrise City (6.50) | no |
| Chill Lofi | Library Rain (5.86) | Library Rain (6.86) | no |
| Deep Intense Rock | Storm Runner (5.88) | Storm Runner (6.86) | no |
| Conflicted | Winter Elegy (4.75) | Winter Elegy (4.55) | no, but margin collapsed |

**Result — more *different* than *more accurate*.** The top pick didn't change for any profile,
which tells me the catalog's genre/mood alignment is strong enough that the winners are robust to
weighting. The real effect was on the **margins and the tail**:

- For *Chill Lofi*, the #3 spot flipped from *Focus Flow* (lofi but "focused") to *Spacewalk
  Thoughts* (ambient but "chill") — halving genre let a cross-genre **mood** match climb.
- For the *Conflicted* profile, the gap between the genre/mood winner and the high-energy
  runner-up shrank from **2.67 points to 0.49** — doubling energy nearly let the song the user's
  *energy* pointed to overtake the song their *genre/mood* pointed to.

**Takeaway:** the `energy` weight is the main lever for how the system resolves conflicts between
"what genre you like" and "what intensity you're in the mood for." I kept the finalized
genre-heavy weights (2.0 / 1.0 / 2.0 / 1.0) as the default, since the experiment didn't improve
the clearly-correct top picks.

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



