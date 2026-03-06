"""
╔══════════════════════════════════════════════════════════════════╗
║  ui/styles.py  —  DataMind Platform                             ║
║  THE COSTUME DESIGNER                                            ║
║                                                                  ║
║  This file controls how EVERYTHING looks in the app.            ║
║  It is written in CSS (Cascading Style Sheets) — the language   ║
║  browsers use to decide colours, sizes, fonts, and animations.  ║
║                                                                  ║
║  We also have Python helper functions that return small pieces  ║
║  of HTML — like building blocks the pages assemble together.    ║
║                                                                  ║
║  Think of it as the wardrobe department of a film set:          ║
║  every actor (element) visits here to get dressed before        ║
║  appearing on screen.                                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
#  PARALLAX_CSS
#  This is one GIANT multi-line string containing all the CSS for the app.
#  In Python, triple quotes """ let a string span many lines.
#  We inject this into Streamlit with st.markdown(PARALLAX_CSS, unsafe_allow_html=True)
#  which tells the browser: "here are your visual instructions".
# ══════════════════════════════════════════════════════════════════════════════

PARALLAX_CSS = """
<style>
/* ════════════════════════════════════════════════════════════════
   CSS COMMENT SYNTAX: everything between /* and */ is a comment.
   CSS rules look like:
     selector { property: value; }
   e.g.  body { background: black; color: white; }
   ════════════════════════════════════════════════════════════════ */

/* ── Google Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;600&display=swap');
/*
  @import fetches external fonts from Google's servers.
  Three fonts are loaded:
  - Orbitron : the futuristic sci-fi font for headings (like in a space movie)
  - Space Mono: a typewriter-style monospace font for data labels
  - Inter    : a clean readable font for body text
  wght@400;700;900 loads three weights: normal, bold, black
*/

/* ── Root Variables ────────────────────────────────────────────── */
:root {
  --void:           #020409;   /* Nearly-black background — the "void of space" */
  --abyss:          #080f1a;   /* Slightly lighter than void */
  --surface:        #0d1b2e;   /* Card/panel background colour */
  --panel:          #0f2035;   /* Slightly lighter panels */
  --border:         #1a3a5c;   /* Border colour for boxes and dividers */
  --cyan:           #00e5ff;   /* The main accent colour: electric cyan */
  --cyan-dim:       #0097a7;   /* Dimmer version of cyan for scrollbars etc. */
  --plasma:         #7c4dff;   /* Deep purple — second accent colour */
  --ember:          #ff6d00;   /* Warm orange — third accent */
  --green:          #00e676;   /* Success green */
  --red:            #ff1744;   /* Error red */
  --text-primary:   #e8f4fd;   /* Main text: near-white with a hint of blue */
  --text-secondary: #7ec8e3;   /* Dimmer text: muted cyan */
  --text-muted:     #3d6680;   /* Even dimmer text: barely visible */
  --glow-cyan:  0 0 20px rgba(0, 229, 255, 0.4), 0 0 60px rgba(0, 229, 255, 0.15);
  --glow-plasma: 0 0 20px rgba(124, 77, 255, 0.4), 0 0 60px rgba(124, 77, 255, 0.15);
  /*
    CSS custom properties (variables). Syntax: --name: value;
    Use them anywhere as var(--name).
    Like named paint pots — change the pot once, everything using it updates.
    --glow-cyan is a box-shadow value: multiple layered glows (inner strong, outer soft).
    rgba(r,g,b,a) = red, green, blue, alpha(opacity). 0.4 = 40% visible.
  */
}

/* ── Global Reset ──────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--void) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}
/*
  html, body, [data-testid="stAppViewContainer"] selects THREE elements at once.
  The comma means "apply this rule to all of these".
  [data-testid="stAppViewContainer"] is an ATTRIBUTE SELECTOR:
    it targets any element that has the attribute data-testid="stAppViewContainer".
    Streamlit gives its main container this attribute.
  !important = "override any other CSS rule, no matter what".
    Like shouting — it always wins. We need it because Streamlit has its own styles.
  var(--void) = use the --void variable we defined in :root above.
*/

/* ── Hide Streamlit chrome ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]    { display: none; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 0 !important; }
/*
  Hide the built-in Streamlit UI elements we don't want:
  #MainMenu = the hamburger menu (top-right of Streamlit apps)
  footer    = the "Made with Streamlit" footer at the bottom
  header    = the top bar Streamlit adds by default
  visibility: hidden = element is invisible but still takes up space
  display: none      = element is completely removed (takes no space)
  .block-container = Streamlit's main content wrapper (class selector = starts with .)
  padding-top: 0 = remove the top gap (Streamlit adds space for its header)
*/

/* ── Scrollbar Styling ─────────────────────────────────────────── */
::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 2px; }
/*
  Style the browser scrollbar (only works in Chrome/Edge, not Firefox).
  ::-webkit-scrollbar  = the scrollbar track + thumb combined
  ::-webkit-scrollbar-track = the background rail of the scrollbar
  ::-webkit-scrollbar-thumb = the draggable handle
  width: 4px = very thin scrollbar (4 pixels wide)
  border-radius: 2px = slightly rounded ends on the thumb
*/

/* ════════════════════════════════════════════════════════════════
   PARALLAX HERO SECTION
   The big full-screen landing section with animated layers.
   "Parallax" = different layers scroll at different speeds,
   creating an illusion of depth — like looking out a train window:
   nearby trees move fast, distant mountains move slowly.
   ════════════════════════════════════════════════════════════════ */

.parallax-hero {
  position: relative;    /* Allow child elements to be positioned relative to this */
  height: 100vh;         /* 100vh = 100% of the viewport (screen) height */
  min-height: 700px;     /* Never smaller than 700px even on tiny screens */
  overflow: hidden;      /* Clip anything that extends outside this box */
  display: flex;         /* Use flexbox to centre the content */
  align-items: center;   /* Vertically centre children */
  justify-content: center; /* Horizontally centre children */
  background: var(--void); /* Very dark background */
}

.parallax-layer {
  position: absolute;  /* Stack all layers on top of each other */
  inset: -20%;         /* Extend 20% BEYOND the parent in all directions */
  /*
    inset: -20% is shorthand for top:-20%; right:-20%; bottom:-20%; left:-20%.
    We make layers BIGGER than their container so when the browser scrolls,
    the background keeps moving without revealing blank edges.
    Like having a painting slightly larger than the frame.
  */
  background-attachment: fixed;
  /*
    KEY PARALLAX TRICK: background-attachment:fixed means the background
    stays fixed relative to the VIEWPORT (the screen), not the element.
    So as you scroll, the background barely moves.
    Different layers scrolling at "different" rates creates the depth effect.
  */
  will-change: transform;
  /*
    will-change: transform hints to the browser: "I'm going to animate this".
    The browser pre-creates a GPU layer for it, making animations silky smooth.
    Like telling a stage crew to pre-position props before the scene.
  */
}

.layer-grid {
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
  /*
    Creates a grid of faint cyan lines.
    First gradient: horizontal lines (0deg = top-to-bottom)
      - 1px of rgba(0,229,255,0.03) = barely visible cyan line
      - then transparent for the rest
    Second gradient: vertical lines (90deg = left-to-right)
      - same pattern at 90 degrees
    Two background-images layered = a grid!
  */
  background-size: 60px 60px;  /* Grid squares are 60px wide and 60px tall */
  animation: gridScroll 30s linear infinite;
  /* Animate continuously (infinite) at constant speed (linear) for 30 seconds per loop */
}

.layer-dots {
  background-image: radial-gradient(circle, rgba(0,229,255,0.12) 1px, transparent 1px);
  /*
    radial-gradient(circle, ...) = a circular gradient.
    Creates tiny 1px cyan dots on a transparent background.
    At 12% opacity they're faint and subtle.
  */
  background-size: 30px 30px;  /* Dots are spaced 30px apart */
  animation: gridScroll 50s linear infinite reverse;
  /* Same animation but slower (50s) and REVERSED direction */
}

.layer-glow-1 {
  background: radial-gradient(ellipse 60% 40% at 20% 50%, rgba(0,229,255,0.08) 0%, transparent 70%);
  /*
    ellipse 60% 40%: an oval shape (60% wide, 40% tall)
    at 20% 50%: positioned at 20% from left, 50% from top
    rgba(0,229,255,0.08): very faint cyan glow at the centre
    0% → transparent 70%: fades from cyan to nothing over 70% of the radius
    This creates a soft "ambient glow" effect in the left-centre area.
  */
  animation: pulse 8s ease-in-out infinite;  /* Gently breathing effect, 8 second cycle */
}

.layer-glow-2 {
  background: radial-gradient(ellipse 50% 60% at 80% 40%, rgba(124,77,255,0.08) 0%, transparent 70%);
  /* Same idea but purple glow in the top-right area */
  animation: pulse 10s ease-in-out infinite reverse;
}

.layer-glow-3 {
  background: radial-gradient(ellipse 30% 30% at 50% 80%, rgba(255,109,0,0.05) 0%, transparent 60%);
  /* Small orange glow at the bottom-centre */
  animation: pulse 12s ease-in-out infinite;
}

/* ── Keyframe Animations ────────────────────────────────────────── */
@keyframes gridScroll {
  from { transform: translate(0, 0); }
  to   { transform: translate(60px, 60px); }
}
/*
  @keyframes defines a named animation sequence.
  "gridScroll" slides the layer diagonally from its original position
  to 60px right and 60px down.
  Because the layer is bigger than its container (inset: -20%),
  this continuous motion looks seamless — you can't see the edges move.
  It creates the illusion of an infinite moving grid.
*/

@keyframes pulse {
  0%, 100% { opacity: 0.5; transform: scale(1);   }
  50%       { opacity: 1;   transform: scale(1.1); }
}
/*
  0% and 100% both start/end at opacity 0.5 and normal size (scale 1).
  50% (halfway) peaks at full opacity and slightly larger (scale 1.1).
  This creates a "breathing" or "heartbeat" effect on the glow layers.
*/

/* ── Hero Content ───────────────────────────────────────────────── */
.hero-content {
  position: relative; /* Sit above the background layers */
  z-index: 10;        /* z-index controls stacking order. 10 = on top of z-index 1,2,3... */
  text-align: center;
  padding: 2rem;
  animation: heroFadeIn 1.2s ease-out forwards;
  /*
    forwards = keep the final state of the animation (don't snap back to "from" state).
    ease-out = starts fast, decelerates to a stop (like a ball rolling to rest).
  */
}

@keyframes heroFadeIn {
  from { opacity: 0; transform: translateY(40px); }  /* Start: invisible, 40px below */
  to   { opacity: 1; transform: translateY(0);    }  /* End: visible, in its natural position */
}
/* The page title "rises in" from below when the page loads. Like a lift arriving. */

.hero-badge {
  display: inline-block;       /* Don't stretch to full width */
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.3em;       /* 0.3em of space between each letter */
  color: var(--cyan);
  background: rgba(0, 229, 255, 0.08);  /* Very faint cyan background */
  border: 1px solid rgba(0, 229, 255, 0.25); /* Thin semi-transparent cyan border */
  padding: 0.4rem 1.2rem;      /* Small vertical padding, larger horizontal */
  border-radius: 2px;          /* Slightly rounded corners */
  margin-bottom: 2rem;
  text-transform: uppercase;   /* Convert text to ALL CAPS automatically */
}

.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(3rem, 8vw, 7rem);
  /*
    clamp(min, preferred, max):
    - Never smaller than 3rem
    - Ideally 8vw (8% of viewport width — so it scales with screen size)
    - Never larger than 7rem
    This makes the title responsive without media queries.
  */
  font-weight: 900;             /* Heaviest weight (black/extra-bold) */
  letter-spacing: -0.02em;     /* Slightly tighter letter spacing for big headings */
  line-height: 0.9;             /* Lines very close together (90% of font size) */
  margin: 0 0 1.5rem;
  background: linear-gradient(135deg, #ffffff 0%, var(--cyan) 50%, var(--plasma) 100%);
  /*
    Gradient: white → cyan → purple, at 135 degrees (diagonal)
    This is applied as a BACKGROUND to the text element...
  */
  -webkit-background-clip: text;  /* ...then CLIPPED to only show through the text shape */
  -webkit-text-fill-color: transparent; /* Make the text itself transparent (shows gradient) */
  background-clip: text;          /* Standard (non-webkit) version of background-clip */
  filter: drop-shadow(0 0 40px rgba(0,229,255,0.3));
  /*
    drop-shadow() is like box-shadow but follows the SHAPE of the content
    (not the rectangular box). Works on transparent elements like our gradient text.
    Creates a soft cyan glow around each letter.
  */
}

.hero-sub {
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  color: var(--text-secondary);
  letter-spacing: 0.05em;
  margin-bottom: 3rem;
  max-width: 600px;              /* Don't let the text get too wide on big screens */
  margin-left: auto;             /* auto margins on both sides centre the element */
  margin-right: auto;
  line-height: 1.8;              /* 1.8 = lines are 180% of font size apart (spacious) */
}

/* ── Floating Particles ─────────────────────────────────────────── */
.particle {
  position: absolute;
  border-radius: 50%;            /* 50% radius = perfect circle */
  animation: floatParticle linear infinite;
  opacity: 0;                    /* Start invisible (animation controls opacity) */
}

@keyframes floatParticle {
  0%   { transform: translateY(100vh) scale(0); opacity: 0; }
  /* Start: below the screen (100vh down), tiny (scale 0), invisible */
  10%  { opacity: 1; }           /* Fade in after 10% of the animation */
  90%  { opacity: 1; }           /* Stay visible until 90% */
  100% { transform: translateY(-10vh) scale(1); opacity: 0; }
  /* End: above the screen (-10vh), full size, invisible again */
}
/* Each particle rises from bottom to top, fading in and out at the edges. */

/* ── Scan Line Effect ───────────────────────────────────────────── */
.scanline {
  position: fixed;               /* fixed = stays in place even when scrolling */
  top: 0; left: 0; right: 0; bottom: 0; /* Cover the ENTIRE screen */
  background: repeating-linear-gradient(
    0deg,
    transparent,               /* Even lines: transparent */
    transparent 2px,
    rgba(0, 0, 0, 0.03) 2px,   /* Odd lines: barely-visible dark stripe */
    rgba(0, 0, 0, 0.03) 4px
  );
  /*
    repeating-linear-gradient: same gradient repeated infinitely.
    Every 4px: 2px transparent, then 2px with 3% dark overlay.
    This creates horizontal "scan lines" like an old CRT monitor or TV screen.
    Very subtle (3% opacity) — gives the retro tech aesthetic.
  */
  pointer-events: none;          /* Clicks pass THROUGH this element (it's just decorative) */
  z-index: 9999;                 /* On top of everything else */
}

/* ── Navigation Bar ─────────────────────────────────────────────── */
.nav-container {
  position: fixed;               /* Sticks to the top of the screen while scrolling */
  top: 0; left: 0; right: 0;    /* Span full width at the top */
  z-index: 1000;
  background: rgba(2, 4, 9, 0.85);  /* 85% opaque dark background */
  backdrop-filter: blur(20px);
  /*
    backdrop-filter: blur() blurs whatever is BEHIND this element.
    Like frosted glass — you can see the content behind but it's blurred.
    20px is quite strong blur.
  */
  border-bottom: 1px solid rgba(0, 229, 255, 0.1); /* Faint cyan bottom border */
  padding: 0 2rem;
  display: flex;
  align-items: center;
  height: 60px;
}

.nav-logo {
  font-family: 'Orbitron', monospace;
  font-weight: 900;
  font-size: 1.1rem;
  color: var(--cyan);
  letter-spacing: 0.1em;
  text-shadow: var(--glow-cyan); /* Apply the cyan glow variable */
}

/* ── Section Styles ─────────────────────────────────────────────── */
.section-header {
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.4em;
  text-transform: uppercase;
  color: var(--cyan);
  margin-bottom: 0.5rem;
}

/* ── Glass Cards ────────────────────────────────────────────────── */
.glass-card {
  background: rgba(13, 27, 46, 0.7);  /* Semi-transparent dark blue */
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.5rem;
  position: relative;     /* Needed for the ::before pseudo-element below */
  overflow: hidden;       /* Clip the glowing top line to this card's shape */
  transition: all 0.3s ease;  /* Smooth all property changes over 0.3 seconds */
}

.glass-card::before {
  /*
    ::before creates a FAKE element BEFORE the card's content.
    It doesn't exist in HTML — CSS generates it.
    We use it to draw the glowing top line without adding HTML.
  */
  content: '';             /* Required for pseudo-elements (even if empty) */
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;             /* Just 1 pixel tall — a thin line across the top */
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  /* Gradient: nothing → cyan → nothing. Creates a glowing centre line. */
  opacity: 0;              /* Start invisible */
  transition: opacity 0.3s ease;
}

.glass-card:hover::before { opacity: 1; }
/* On hover: reveal the glowing top line */
.glass-card:hover {
  border-color: rgba(0, 229, 255, 0.3);
  box-shadow: 0 0 30px rgba(0, 229, 255, 0.05);
}
/* :hover = applies when the mouse is over this element */

/* ── Metric Cards ───────────────────────────────────────────────── */
.metric-card {
  background: linear-gradient(135deg, rgba(13,27,46,0.9), rgba(8,15,26,0.95));
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.2rem 1.5rem;
  position: relative;
}

.metric-card .metric-value {
  /*
    ".metric-card .metric-value" = selects .metric-value elements
    that are INSIDE .metric-card elements. Descendant combinator.
  */
  font-family: 'Orbitron', monospace;
  font-size: 2rem;
  font-weight: 900;
  color: var(--cyan);
  text-shadow: var(--glow-cyan);
}

.metric-card .metric-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

.metric-card::after {
  /*
    ::after is like ::before but creates an element AFTER the content.
    Here we use it to draw a coloured line at the BOTTOM of the card.
  */
  content: '';
  position: absolute;
  bottom: 0; left: 0;
  width: 40%;       /* Only 40% of the card width — like a partial underline */
  height: 2px;
  background: var(--cyan);
  box-shadow: var(--glow-cyan);
}

/* ── Terminal Box ───────────────────────────────────────────────── */
.terminal-box {
  background: rgba(2, 4, 9, 0.95);
  border: 1px solid var(--border);
  border-radius: 4px;
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  padding: 1.5rem;
  min-height: 200px;
  max-height: 500px;
  overflow-y: auto;   /* Add scrollbar if content is taller than 500px */
  line-height: 1.8;
}
/* .terminal-box looks like a hacker terminal: dark, monospace font, barely any border */

.terminal-output {
  color: var(--text-secondary);
  padding-left: 1rem;
  border-left: 2px solid rgba(0, 229, 255, 0.2);
  /* A faint cyan left border — like a quote block but for AI output */
  margin: 0.5rem 0;
}

/* ── Chat Messages ──────────────────────────────────────────────── */
.msg-user {
  background: rgba(0, 229, 255, 0.06);
  border: 1px solid rgba(0, 229, 255, 0.15);
  border-radius: 4px 4px 0 4px;
  /*
    border-radius accepts up to 4 values: top-left, top-right, bottom-right, bottom-left.
    4px 4px 0 4px = rounded on 3 corners, FLAT on bottom-right.
    This creates the "speech bubble" shape pointing to the bottom-right (user side).
  */
  padding: 0.8rem 1.2rem;
  margin: 0.5rem 0;
  max-width: 80%;       /* Never wider than 80% of the container */
  margin-left: auto;    /* auto left margin pushes it to the RIGHT (user messages align right) */
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  color: var(--text-primary);
}

.msg-ai {
  background: rgba(124, 77, 255, 0.06);    /* Very faint purple tint */
  border: 1px solid rgba(124, 77, 255, 0.15);
  border-radius: 4px 4px 4px 0;
  /* Flat on bottom-LEFT — speech bubble points to bottom-left (AI side) */
  padding: 0.8rem 1.2rem;
  margin: 0.5rem 0;
  max-width: 90%;      /* AI messages can be a bit wider */
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.msg-label {
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
  opacity: 0.6;          /* 60% opaque = slightly faded label */
}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button {
  /*
    .stButton is the wrapper Streamlit adds around every st.button().
    > means "direct child" (immediately inside, not nested deeper).
    So this targets the actual <button> element inside Streamlit's wrapper.
  */
  background: transparent !important;
  border: 1px solid var(--cyan) !important;
  color: var(--cyan) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  padding: 0.6rem 1.5rem !important;
  border-radius: 2px !important;
  transition: all 0.3s ease !important;
  /* transition: all means every property smoothly animates on change */
}

.stButton > button:hover {
  background: rgba(0, 229, 255, 0.1) !important;  /* Faint cyan fill on hover */
  box-shadow: var(--glow-cyan) !important;          /* Glow effect on hover */
}

/* ── Text Inputs ────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  /* Deep selector chain: Streamlit wraps inputs in multiple nested divs */
  background: rgba(8, 15, 26, 0.9) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
  border-radius: 2px !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  /* :focus = when the user clicks into the input to type */
  border-color: var(--cyan) !important;
  box-shadow: 0 0 10px rgba(0, 229, 255, 0.2) !important;
  /* Subtle glow when the field is active */
}

/* ── Select Boxes ───────────────────────────────────────────────── */
.stSelectbox > div > div {
  background: rgba(8, 15, 26, 0.9) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  border-radius: 2px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(4, 8, 15, 0.98) !important;    /* Near-opaque very dark blue */
  border-right: 1px solid var(--border) !important; /* Thin border on the right edge */
}

[data-testid="stSidebar"] .stRadio label {
  /* .stRadio label = label text inside a radio button group in the sidebar */
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--text-secondary) !important;
}

/* ── Data Tables ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
}

/* ── Progress Bars ──────────────────────────────────────────────── */
.stProgress > div > div > div > div {
  /* Very deep nesting — Streamlit's progress bar has many wrapper divs */
  background: linear-gradient(90deg, var(--cyan), var(--plasma)) !important;
  /* Gradient bar: cyan on the left → purple on the right */
}

/* ── Tabs ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  /* data-baseweb is a React UI library attribute Streamlit uses internally */
  background: transparent !important;
  border-bottom: 1px solid var(--border);
  gap: 0;   /* No gaps between tabs */
}

.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.1em !important;
  border-radius: 0 !important;         /* Square tabs (no rounding) */
  padding: 0.8rem 1.5rem !important;
}

.stTabs [aria-selected="true"] {
  /* aria-selected="true" is the currently ACTIVE tab (screen reader accessible) */
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;     /* Cyan underline on active tab */
  background: rgba(0, 229, 255, 0.05) !important;      /* Very faint cyan background */
}

/* ── Status Indicator (Online/Offline dot) ──────────────────────── */
.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;           /* Circle */
  margin-right: 0.5rem;
  animation: blink 2s ease-in-out infinite; /* Gently pulsing dot */
}

.status-dot.online  { background: var(--green); box-shadow: 0 0 8px var(--green); }
/* .status-dot.online = element that has BOTH classes: status-dot AND online */
.status-dot.offline { background: var(--red);   box-shadow: 0 0 8px var(--red);   }

@keyframes blink {
  0%, 100% { opacity: 1;   }
  50%       { opacity: 0.4; }
}
/* Gently dims to 40% opacity at the halfway point, creating a "breathing" effect */

/* ── Feature Cards ──────────────────────────────────────────────── */
.feature-card {
  background: linear-gradient(135deg, rgba(13,27,46,0.8), rgba(8,15,26,0.9));
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
  cursor: pointer;        /* Shows a hand cursor on hover (like a link) */
  transition: all 0.4s ease;
}

.feature-card:hover {
  transform: translateY(-4px);   /* Lift up 4px on hover (feels clickable) */
  border-color: var(--cyan);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 229, 255, 0.1);
  /* Two shadows: a dark drop shadow for depth, plus a cyan glow */
}

.feature-card .accent-line {
  position: absolute;
  bottom: 0; left: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--plasma));
  transition: width 0.4s ease;
  width: 0;                /* Start with zero width (hidden) */
}
.feature-card:hover .accent-line { width: 100%; }
/* On hover: animate the accent line from 0 to 100% width — slides in from left */

/* ── Stat Bars ──────────────────────────────────────────────────── */
.stat-bar {
  height: 3px;
  background: rgba(255,255,255,0.05); /* Faint background track */
  border-radius: 2px;
  overflow: hidden;
}

.stat-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--cyan), var(--plasma));
  transition: width 1s ease;  /* Width animates over 1 second when changed */
  box-shadow: 0 0 8px rgba(0, 229, 255, 0.5);
}

/* ── Expander ───────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--text-secondary) !important;
  background: rgba(8, 15, 26, 0.5) !important;
  border: 1px solid var(--border) !important;
}

/* ── Alerts / Info Boxes ────────────────────────────────────────── */
[data-testid="stInfo"] {
  background: rgba(0, 229, 255, 0.05) !important;
  border: 1px solid rgba(0, 229, 255, 0.2) !important;
  color: var(--text-secondary) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
}

[data-testid="stSuccess"] {
  background: rgba(0, 230, 118, 0.05) !important;
  border: 1px solid rgba(0, 230, 118, 0.2) !important;
}

[data-testid="stError"] {
  background: rgba(255, 23, 68, 0.05) !important;
  border: 1px solid rgba(255, 23, 68, 0.2) !important;
}

/* ── Spinner ────────────────────────────────────────────────────── */
.stSpinner > div {
  border-color: var(--cyan) transparent transparent transparent !important;
  /*
    A spinner is a circle with only ONE coloured side.
    top = cyan, right = transparent, bottom = transparent, left = transparent.
    When it rotates, the cyan part spins around, creating a "loading" ring effect.
  */
}

/* ── File Uploader ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--border) !important;  /* Dashed border = "drop here" signal */
  background: rgba(8, 15, 26, 0.5) !important;
  border-radius: 4px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--cyan) !important;  /* Highlight when hovering to drag a file in */
}

/* ── Slider ─────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
  /* role="slider" is the draggable thumb of the slider */
  background: var(--cyan) !important;
  box-shadow: var(--glow-cyan) !important;
}

/* ── Checkbox ───────────────────────────────────────────────────── */
.stCheckbox > label > div:first-child {
  border-color: var(--border) !important;
  background: transparent !important;
}
.stCheckbox > label > div:first-child[aria-checked="true"] {
  background: var(--cyan) !important;
  border-color: var(--cyan) !important;
  /* Checked state: fill the box with cyan */
}

/* ── Plotly Chart Background ────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg {
  background: transparent !important;
  /* Remove Plotly's default white background so our dark theme shows through */
}

/* ── Loading Shimmer ────────────────────────────────────────────── */
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position:  200% 0; }
}
.skeleton {
  background: linear-gradient(90deg,
    rgba(26,58,92,0.3) 25%,    /* Dark */
    rgba(0,229,255,0.05) 50%,  /* Brief highlight */
    rgba(26,58,92,0.3) 75%     /* Dark again */
  );
  background-size: 200% 100%;  /* Make gradient twice as wide as the element */
  animation: shimmer 2s infinite;
  border-radius: 4px;
  /*
    The shimmer "slides" the 200%-wide gradient horizontally.
    The bright stripe moves left-to-right, creating a loading animation.
    Like a metallic sheen sliding across a loading bar.
  */
}

/* ── Corner Bracket Decoration ──────────────────────────────────── */
.bracket-box { position: relative; padding: 1.5rem; }
.bracket-box::before, .bracket-box::after {
  content: '';
  position: absolute;
  width: 20px;
  height: 20px;
}
.bracket-box::before {
  top: 0; left: 0;
  border-top: 2px solid var(--cyan);
  border-left: 2px solid var(--cyan);
  /* Top-left corner bracket [ */
}
.bracket-box::after {
  bottom: 0; right: 0;
  border-bottom: 2px solid var(--cyan);
  border-right: 2px solid var(--cyan);
  /* Bottom-right corner bracket ] */
}
/* Together these two pseudo-elements create ⌐ and ¬ corner brackets on the box */

</style>

<div class="scanline"></div>
<!-- The scanline overlay div — covers the whole screen with subtle horizontal lines.
     Defined once here in the CSS string, rendered on every page. -->
"""


# ══════════════════════════════════════════════════════════════════════════════
#  PYTHON HELPER FUNCTIONS
#  Each function returns a string of HTML.
#  The pages call these functions and inject the HTML with st.markdown(..., unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════

def hero_html():
    """
    Returns the HTML string for the full-screen parallax landing page hero.
    Contains 5 stacked parallax layers + floating particles + the title text.

    NOTE: No HTML comments (<!-- -->) inside this string.
    Streamlit's markdown renderer can sometimes display HTML comment text
    as visible content, so we keep comments only in Python (# style).
    """
    return """
<div class="parallax-hero">
  <div class="parallax-layer layer-grid"></div>
  <div class="parallax-layer layer-dots"></div>
  <div class="parallax-layer layer-glow-1"></div>
  <div class="parallax-layer layer-glow-2"></div>
  <div class="parallax-layer layer-glow-3"></div>

  <div class="particle" style="left:10%;width:3px;height:3px;background:#00e5ff;animation-duration:12s;animation-delay:0s;"></div>
  <div class="particle" style="left:25%;width:2px;height:2px;background:#7c4dff;animation-duration:18s;animation-delay:3s;"></div>
  <div class="particle" style="left:40%;width:4px;height:4px;background:#00e5ff;animation-duration:15s;animation-delay:6s;"></div>
  <div class="particle" style="left:60%;width:2px;height:2px;background:#ff6d00;animation-duration:20s;animation-delay:1s;"></div>
  <div class="particle" style="left:75%;width:3px;height:3px;background:#7c4dff;animation-duration:14s;animation-delay:4s;"></div>
  <div class="particle" style="left:90%;width:2px;height:2px;background:#00e5ff;animation-duration:16s;animation-delay:8s;"></div>

  <div class="hero-content">
    <div class="hero-badge">&#11041; Intelligent Data Science Platform &#11041;</div>
    <h1 class="hero-title">DATA<br/>MIND</h1>
    <p class="hero-sub">
      Multi-Agent AI &nbsp;&middot;&nbsp; RAG Pipeline &nbsp;&middot;&nbsp; AutoML &nbsp;&middot;&nbsp; NL Queries<br/>
      Powered by local LLMs via Ollama &nbsp;&middot;&nbsp; Zero Cloud Cost
    </p>
  </div>
</div>
"""


def section_divider(text=""):
    """
    Returns a horizontal divider line with a text label centred in the middle.
    e.g. ────────── VISUALIZATIONS ──────────
    """
    return f"""
<div style="display:flex;align-items:center;gap:1rem;margin:2rem 0;">
  <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(26,58,92,0.8));"></div>
  <!--
    flex:1 = this div takes up all available space (it's flexible).
    height:1px = it's just a thin line.
    gradient: transparent on the left, fading to border colour on the right.
    Left line fades IN from the left.
  -->

  <span style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:0.3em;
               text-transform:uppercase;color:rgba(61,102,128,0.8);">{text}</span>
  <!-- {text} is replaced by the Python f-string with the actual label text -->

  <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(26,58,92,0.8),transparent);"></div>
  <!-- Right line fades OUT to the right. Mirror of the left line. -->
</div>
"""


def metric_card_html(value, label, suffix="", color="var(--cyan)"):
    """
    Returns a styled stats card showing a big number and a label.
    e.g.  891      or    12 MB
          ROWS           MEMORY
    """
    return f"""
<div class="metric-card" style="--accent:{color};">
  <!--
    style="--accent:{color}" sets a CSS custom property on THIS element.
    Child elements can use var(--accent) to inherit this colour.
    It's like passing a colour argument through CSS.
  -->
  <div class="metric-value" style="color:{color};text-shadow:0 0 20px {color}40;">{value}{suffix}</div>
  <!--
    {color}40 adds "40" after the hex colour code = 40 in hex = 64 in decimal = 25% opacity.
    So if color is #00e5ff, the shadow colour is #00e5ff40 (25% transparent cyan).
  -->
  <div class="metric-label">{label}</div>
</div>
"""


def status_badge(text, online=True):
    """
    Returns HTML for a small status badge with a blinking dot.
    green dot = Ollama Online   red dot = Ollama Offline
    """
    cls = "online" if online else "offline"
    # cls = "online" or "offline" — this becomes the CSS class on the dot.
    return (
        f'<span>'
        f'<span class="status-dot {cls}"></span>'
        # The blinking dot — CSS class gives it green or red colour + glow.
        f'<span style="font-family:\'Space Mono\',monospace;font-size:0.7rem;'
        f'color:var(--text-secondary);">{text}</span>'
        # The text next to the dot.
        f'</span>'
    )
    # Note: \' inside f-strings escapes the single quote so it doesn't end the string.


def feature_card_html(icon, title, desc, accent="var(--cyan)"):
    """
    Returns HTML for a feature showcase card with icon, title, description,
    and a coloured hover accent line at the bottom.
    """
    return f"""
<div class="feature-card">
  <span class="feature-icon">{icon}</span>    <!-- e.g. 🔬 -->
  <h3>{title}</h3>
  <p>{desc}</p>
  <div class="accent-line" style="background:linear-gradient(90deg,{accent},var(--plasma));"></div>
  <!-- accent-line slides in from left on hover (CSS handles the animation) -->
</div>
"""


def sidebar_logo():
    """
    Returns the HTML for the DataMind logo at the top of the sidebar.
    DATAMIND in two colours: cyan for DATA, purple for MIND.
    """
    return """
<div style="padding:1.5rem 1rem 1rem;border-bottom:1px solid rgba(26,58,92,0.5);margin-bottom:1rem;">
  <div style="font-family:'Orbitron',monospace;font-size:1.2rem;font-weight:900;
              color:#00e5ff;text-shadow:0 0 20px rgba(0,229,255,0.4);letter-spacing:0.1em;">
    DATA<span style="color:#7c4dff;">MIND</span>
    <!--
      <span> is an inline element — it wraps text without starting a new line.
      Here it only changes the colour of "MIND" to purple, leaving "DATA" cyan.
      The <span> is INSIDE the larger div so font/size are inherited.
    -->
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:0.3em;
              text-transform:uppercase;color:rgba(61,102,128,0.8);margin-top:0.3rem;">
    AI Platform v2.0
  </div>
</div>
"""
