# B.3-stretch — Grounding the ledger in real incidents

Five narratives pulled from NASA's Aviation Safety Reporting System (ASRS) CALLBACK
newsletter, an official NASA publication that digests real, submitted ASRS reports.
ASRS does not attach public ACN numbers to these digest narratives (only to raw
database search results), so citations here are by CALLBACK issue, title, and date —
each is a traceable, real incident, not a paraphrase from memory.

All five share the exact failure mode the simulated ledger calls `over_reliance`:
a skilled operator deferred to automation specifically *because* it had been reliable
up to that point, and missed something a fresh, unaided look would have caught.

---

**1. "Automating Complacency" — Gulfstream V, CALLBACK Issue 446 (March 2017)**
https://asrs.arc.nasa.gov/publications/callback/cb_446.html

A pilot let the autopilot capture an assigned altitude of 14,000 ft while distracted
reviewing arrival information. The aircraft instead continued descending to 13,300 ft
before anyone caught it. In his own words:

> "I got complacent…, and I believe it was because for so many years of operating
> this equipment, never had the automation failed to perform as it had been set up."

This is the cleanest match in the set to `over_reliance` as defined in this project's
B.1 taxonomy: the deferral itself wasn't unreasonable given the track record — it was
wrong on *this* case, and years of correct automation is exactly what made the miss
possible.

**2. "How Low Should You Go?" — B737, CALLBACK Issue 440 (September 2016)**
https://asrs.arc.nasa.gov/publications/callback/cb_440.html

A Captain set the STAR's lowest published altitude and let VNAV manage the rest,
skipping the manual habit of stepping down through each restriction. The aircraft
descended below the mandatory 10,000 ft crossing altitude before the crew caught it:

> "I had been relying on the VNAV automation instead of the old fashioned, 'Set the
> next lowest altitude,' which forces both pilots [to be] situationally aware."

This one names the trade-off directly — the manual habit is described as the thing
that *forces* awareness, and automation is what let that habit atrophy. It's the
aviation-domain version of this project's core question: does routing decisions
through AI quietly retire the independent check that used to catch errors?

**3. "More Than Meets the Eye" — B737, CALLBACK Issue 440 (September 2016)**
https://asrs.arc.nasa.gov/publications/callback/cb_440.html

A crew let automation calculate descent timing without cross-checking it against
wind, arriving high at a crossing restriction:

> "I was trusting in the automation too much for when to start my descent."

**4. "Teetering on the Approach" — Gulfstream, CALLBACK Issue 440 (September 2016)**
https://asrs.arc.nasa.gov/publications/callback/cb_440.html

On an ILS approach into Teterboro in strong crosswinds, the Captain fixated on
*watching the autopilot correct* rather than monitoring the aircraft's actual
position, and drifted left of course:

> "The PIC's comment was, 'Look at how much correction this thing is putting in.'
> We continued to drift left."

Distinct sub-mode of the same failure: not blind deferral, but attention captured by
monitoring the automation itself, at the cost of monitoring the thing the automation
was supposed to be helping with.

**5. "We Were Supposed To Be Descending" — B737-700, CALLBACK Issue 407 (December 2013)**
https://asrs.arc.nasa.gov/publications/callback/cb_407.html

A Captain attempted to engage VNAV for a descent; it didn't engage. He believed the
descent was happening anyway and didn't notice the aircraft had climbed from 27,600 ft
to 30,000 ft until leveling off. The expectation set by normal automation behavior
overrode what the instruments were actually showing.

---

## Tying back to the simulated ledger

The simulated `over_reliance` rate in this project's B.1 output is not a rare edge
case — it runs 5.5%–8.0% of all `ai_shown=1` events depending on arm (`out/
reliance_by_arm.csv`), and it's exactly the `control_always_ai` arm — the one with
zero forced independent-judgment practice — that carries both the highest simulated
`over_reliance` rate (8.0%) and, per `out/ledger_benefit_by_arm.csv`, the steepest
real capability decline (mean change −0.268 vs. −0.018 for `blind_first`). These five
ASRS narratives are the qualitative version of that same number: skilled operators,
reliable automation, and a miss that specifically required the automation to have
been trustworthy for a long time first. The simulation says this pattern is common
and quantifiable; the incident record says it's also exactly what it looks like when
it happens to a real person at the controls of a real aircraft.
