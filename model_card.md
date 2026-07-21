# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**VibeScope 1.0** — it scopes out songs that match your vibe.

---

## 2. Intended Use  

**Goal / Task.** VibeScope takes one person's stated music taste and suggests songs from a
small catalog that best match it. It tries to predict which songs "feel right" for a user
based on their favorite genre, favorite mood, preferred energy level, and whether they like
acoustic music.

**Who it's for.** This is a classroom project for learning how recommenders work. It is not
built for real users or real listening apps.

**Assumptions it makes.** It assumes the user can describe their taste in a few simple values,
and that those values stay the same for the whole session. It has no idea what the user has
listened to before.

**Intended use:**

- Learning and demonstrating how content-based scoring turns data into recommendations.
- Experimenting with weights to see how rankings change.

**Non-intended use:**

- Real product recommendations for real listeners.
- Any decision that matters (it uses only 18 songs and hand-picked weights).
- Judging the quality or worth of any song or artist.

---

## 3. How the Model Works  

**Algorithm summary (plain language).** The model gives every song a score, then sorts the
songs from highest to lowest and shows the top 5.

Each song can earn points in four ways:

- **Genre match:** +2 points if the song's genre is the user's favorite.
- **Mood match:** +1 point if the song's mood is the user's favorite.
- **Energy closeness:** up to +2 points for being close to the energy level the user wants.
  A perfect match gets the full 2 points; the farther off it is, the fewer points it gets.
- **Acoustic preference:** up to +1 point. If the user likes acoustic music, acoustic songs
  score higher; if not, it's the reverse.

The most a song can score is **6 points**. Genre is worth the most on purpose, so
recommendations stay close to the user's favorite style.

**What each recommendation tells you.** Along with the score, the model lists the reasons a song
was picked (like "genre match (+2.0)"), so it's easy to see *why* it was chosen.

**Changes from the starter code.** The starter files were empty stubs. I wrote the song loader,
the scoring rules, and the ranking logic, expanded the catalog from 10 to 18 songs, and moved
the weights into named settings so they're easy to experiment with.

---

## 4. Data  

**Size.** The catalog has **18 songs**, stored in `data/songs.csv`.

**Features per song.** Each song has an id, title, artist, genre, mood, and five numeric
values: energy, tempo, valence (how positive it sounds), danceability, and acousticness.
Right now the scoring uses genre, mood, energy, and acousticness.

**Genres represented.** pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip hop,
classical, edm, country, metal, reggae, r&b, and folk.

**Moods represented.** happy, chill, intense, relaxed, moody, focused, energetic, melancholy,
euphoric, nostalgic, aggressive, uplifting, romantic, and dreamy.

**Changes I made.** The starter file had 10 songs. I added 8 more to cover genres and moods
that were missing, so different users would get more varied results.

**What's missing.** The catalog is tiny, so many genres appear only once or twice. It has no
lyrics, no language info, and no real listening history. It also can't capture personal or
situational taste — like a song being your "late-night driving" song.

---

## 5. Strengths  

- **Clear, "typical" tastes work great.** For users like High-Energy Pop, Chill Lofi, and Deep
  Intense Rock, the top pick matched my own gut feeling every time.
- **Every result is explained.** The model shows the exact points a song earned, so you can
  always see why it was chosen. Nothing is a black box.
- **It captures the energy vibe well.** Songs close to the user's energy level rise to the top,
  which lines up with how people actually pick calm vs. hyped music.
- **It fails gracefully.** Even when a user asks for a genre or mood that doesn't exist, the
  model still returns sensible songs instead of crashing or returning nothing.

---

## 6. Limitations and Bias 

The clearest weakness I found is a **genre-heavy bias that can override what the user
actually asked for on other features.** Because an exact genre match is worth 2.0 points
and a mood match 1.0, the combined +3.0 from categorical matches can outweigh a large
penalty on energy. I saw this directly in my "Conflicted" test profile, which asked for
*high energy (0.9)* but a *melancholy* mood: the system confidently recommended a slow,
low-energy classical piece (energy 0.30) simply because it matched the genre and mood, even
though its energy was the opposite of what the user requested. A second, related limitation
is a **small-catalog filter bubble**: some genres appear only once (one rock song, one metal
song), so for a rock fan the top result is correct but the rest of the list is padded with
unrelated high-energy songs from other genres — there simply isn't enough variety to fill a
good top-5. Finally, the **energy gap is measured linearly** (`1 - |song - target|`) and every
numeric feature is treated as equally spaced, so the model has no notion that, say, a 0.1
energy difference matters more for calm music than for intense music. Together these mean the
system quietly favors users whose taste is concentrated in a well-represented genre and
penalizes users with unusual or cross-genre tastes.

---

## 7. Evaluation  

**Profiles tested.** I ran five user profiles through the recommender: three ordinary tastes —
*High-Energy Pop*, *Chill Lofi*, and *Deep Intense Rock* — and two "adversarial" profiles built
to probe the logic: *Conflicted* (high energy but a melancholy mood) and *Impossible* (a genre
and mood that don't exist in the catalog). For each I looked at the top 5 songs and the reason
each was chosen.

**What surprised me.** The top pick for all three normal profiles felt exactly right and matched
my intuition, which was reassuring. The surprise came from the adversarial profiles: the
*Conflicted* profile exposed that genre/mood matching can beat energy entirely (it recommended a
slow classical song to someone who wanted high energy), and the *Impossible* profile surprised me
in a good way — instead of crashing or returning nothing, it fell back cleanly to the numeric
features and still returned sensible mid-energy songs.

**Comparing the profiles (what changed and why it makes sense):**

- **High-Energy Pop vs. Chill Lofi:** near-opposites. Pop pulls loud, non-acoustic, upbeat songs
  (Sunrise City, Gym Hero); Lofi shifts to quiet, acoustic, low-energy tracks (Library Rain,
  Midnight Coding). This is the clearest sign the profiles actually test different things.
- **High-Energy Pop vs. Deep Intense Rock:** both want high energy and non-acoustic, so they
  *share* songs like Gym Hero and Pulse Machine — but the #1 differs (Sunrise City vs. Storm
  Runner) purely because of the genre match. Same energy, different genre anchor.
- **Chill Lofi vs. Deep Intense Rock:** the biggest contrast. One wants calm/acoustic, the other
  loud/electric, and their top-5 lists share zero songs — exactly what you'd hope.
- **Conflicted vs. any normal profile:** normal profiles produce a confident winner far ahead of
  the pack; the Conflicted profile produces a much tighter race (the top scores bunch together)
  because the song's features pull in opposite directions. The scoring recipe resolves the tie by
  siding with genre + mood.
- **Impossible vs. every other profile:** it's the only profile where *no* song earns genre or
  mood points, so all scores are low (≤ 2.5) and ranking is decided entirely by energy and
  acousticness. It behaves like a "numeric-only" version of the recommender.

**Plain-language explanation (why "Gym Hero" keeps showing up for "Happy Pop"):** Imagine you tell
a friend you want happy pop songs with lots of energy. Every song gets points for three things:
being pop, feeling happy, and having the energy level you asked for. "Gym Hero" is a pop song with
very high energy — so it scores big on *pop* and *energy* even though its mood is "intense," not
"happy." It loses the happy-mood points but still beats most songs on the other two, so it keeps
landing near the top. In short, the song wins on two out of three things you care about, and that's
usually enough to rank high.

---

## 8. Future Work  

Three things I'd change if I kept building this:

1. **Grow the catalog.** With more songs per genre, the top-5 lists would have real variety
   instead of padding the tail with unrelated songs.
2. **Use more features and let users set them.** The profile only has four knobs. Adding
   valence, danceability, and tempo as user preferences would capture taste more finely.
3. **Add a diversity rule.** Right now the top results can all be the same genre. I'd add a
   step that mixes in a few different-but-close songs so users discover new styles.

---

## 9. Personal Reflection  

**Biggest learning moment.** My biggest "aha" was realizing that a recommendation is really
just sorting. Once each song has a score, recommending is nothing more than lining them up and
taking the top few. Seeing that the whole system boils down to a scoring rule plus a ranking
rule made recommenders feel a lot less magical and a lot more understandable.

**How AI tools helped, and when I double-checked them.** The AI helped me move fast — brainstorming
the scoring formula, writing the CSV loader, and cleaning up the terminal output. But I had to
double-check its choices. For example, the weights are a judgment call, not a fact, so I decided
genre vs. mood myself instead of trusting a default. I also caught a real bug it surfaced: the
starter code imported `recommender` one way but the tests imported `src.recommender`, and I had to
pick one convention. The lesson was that AI is great for drafts, but the logic and the values
behind it still need to be my decision.

**What surprised me about simple algorithms.** I was surprised that such basic math — adding a few
points and sorting — actually "feels" like a real recommendation. When the Chill Lofi profile
returned a perfect low-energy match, it felt smart, even though there's no learning or intelligence
behind it at all. It made me realize a lot of what feels like "smart" tech is just clear rules
applied consistently.

**What I'd try next.** I'd grow the catalog, let users express more of their taste than four
values, and add a diversity rule so the top results aren't all the same genre. I'd also like to
test the system against real people to see if the scores match what they'd actually pick.
