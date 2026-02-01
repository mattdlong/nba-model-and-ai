# NBA Betting Model - Complete Usage Guide

A friendly guide for people who want to understand and use this NBA game prediction system, even if you've never written code or placed a bet before.

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [How Does It Work?](#how-does-it-work)
3. [Getting Started](#getting-started)
4. [Using the System](#using-the-system)
5. [Understanding the Components](#understanding-the-components)
6. [Glossary](#glossary)

---

## What Is This?

### The Simple Explanation

Imagine you have a really smart friend who watches every NBA game, remembers every player's performance, knows how tired each team is, and can spot patterns humans miss. This system is like that friend - but it's a computer program.

**What it does:** Predicts which team will win NBA basketball games and suggests smart betting decisions.

**Why it exists:** Sports betting markets aren't perfect. Sometimes the odds offered by sportsbooks don't reflect the true probability of an outcome. This system tries to find those opportunities.

**How it helps:** Instead of guessing or going with your gut, you get predictions backed by data from thousands of games, player statistics, and advanced mathematics.

### Think of It Like...

- **A Weather Forecast for Basketball:** Just like meteorologists use data to predict rain, this system uses basketball data to predict game outcomes. It won't be right 100% of the time, but it's more reliable than guessing.

- **A Financial Advisor for Sports Bets:** Instead of betting randomly, the system tells you how much to bet based on how confident it is - just like a financial advisor balances risk and reward.

---

## How Does It Work?

### The Big Picture (5-Step Process)

```
Step 1: COLLECT DATA
        ↓
        Gather information about every NBA game, player, and play

Step 2: BUILD FEATURES
        ↓
        Turn raw data into useful insights (like "how good is this player really?")

Step 3: TRAIN THE BRAIN
        ↓
        Teach the computer to recognize patterns that predict wins

Step 4: MAKE PREDICTIONS
        ↓
        For upcoming games, estimate who will win and by how much

Step 5: GENERATE SIGNALS
        ↓
        Compare predictions to betting odds and suggest smart bets
```

### A Real-World Analogy

Think of it like preparing for a job interview:

1. **Collect Data** = Research the company (gather all available information)
2. **Build Features** = Summarize key points (turn information into useful insights)
3. **Train the Brain** = Practice with mock interviews (learn from past experiences)
4. **Make Predictions** = Anticipate questions (predict what will happen)
5. **Generate Signals** = Decide what to emphasize (make strategic decisions)

---

## Getting Started

### What You Need Before Starting

1. **A Computer** - Mac, Windows, or Linux
2. **Python 3.11 or newer** - A programming language (free to download)
3. **About 5GB of storage** - For the database of NBA games
4. **Internet connection** - To download NBA data

### Installation (Copy-Paste These Commands)

Open your terminal (Mac: Terminal app, Windows: Command Prompt or PowerShell) and run:

```bash
# Step 1: Go to the project folder
cd ~/Documents/code/nba-model-and-ai

# Step 2: Create a virtual environment (a clean workspace)
python -m venv .venv

# Step 3: Activate the environment
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Step 4: Install the required packages
pip install -e ".[dev]"

# Step 5: Set up the database
python -m nba_model.cli data collect --seasons 2023-24
```

### Verify It's Working

```bash
# Check the CLI is working
python -m nba_model.cli --help

# Check database status
python -m nba_model.cli data status
```

You should see a list of commands and database statistics.

---

## Using the System

### Daily Workflow (What Most Users Do)

```bash
# 1. Update with yesterday's results
python -m nba_model.cli data update

# 2. Get today's predictions
python -m nba_model.cli predict today

# 3. See betting signals (recommended bets)
python -m nba_model.cli predict signals
```

### Example Output

When you run `predict today`, you might see:

```
Today's Games (January 15, 2024)
================================

LAL @ BOS
  Home Win Probability: 62%
  Predicted Margin: +5.5 (Boston by 5.5 points)
  Predicted Total: 223 points
  Confidence: High

MIA @ NYK
  Home Win Probability: 48%
  Predicted Margin: -2.0 (Miami by 2 points)
  Predicted Total: 211 points
  Confidence: Medium
```

When you run `predict signals`, you might see:

```
Betting Signals
===============

SIGNAL: Boston Celtics Moneyline
  Game: LAL @ BOS
  Model Says: 62% chance Boston wins
  Market Says: 55% chance (odds: -120)
  Edge: +7% (model thinks Boston is undervalued)
  Suggested Bet: 1.5% of bankroll
  Confidence: High

No signal for MIA @ NYK (edge too small)
```

### All Available Commands

| Command | What It Does | When to Use It |
|---------|-------------|----------------|
| `data collect` | Download NBA game data | First-time setup or adding new seasons |
| `data update` | Get yesterday's results | Daily, before making predictions |
| `data status` | Show database statistics | Check what data you have |
| `features build` | Calculate player/team metrics | After collecting new data |
| `train all` | Train the prediction models | Weekly or when triggered |
| `backtest run` | Test model on historical data | After training to verify performance |
| `monitor drift` | Check if model is degrading | Daily, catches problems early |
| `predict today` | Predictions for today's games | Daily, before games start |
| `predict game GAMEID` | Prediction for one game | When you want details on a specific game |
| `predict signals` | Betting recommendations | When you want to see where to bet |
| `dashboard build` | Create web dashboard | When you want to view results in browser |

---

## Understanding the Components

This section explains each part of the system in detail. Each component is like a department in a company - they work together to produce the final result.

---

### Component 1: Data Collection (`nba_model/data/`)

#### What Is It?

The "research department" that gathers all the raw information about NBA games.

#### What Does It Do?

- Downloads game schedules and scores
- Collects player statistics (points, rebounds, assists, etc.)
- Gets play-by-play data (every event that happens in a game)
- Gathers shot location data (where players shoot from)
- Stores everything in an organized database

#### Why Does It Matter?

Without good data, predictions are just guesses. This component ensures we have comprehensive, accurate information about every NBA game.

#### Key Functions

**`NBAApiClient`** - The Data Fetcher

*What it does:* Connects to the NBA's data service and downloads information.

*Think of it like:* A librarian who knows exactly where to find any book you need.

```python
# Example: Get all games from a season
from nba_model.data import NBAApiClient

client = NBAApiClient()
games = client.get_league_game_finder(season="2023-24")
# Returns: A table with every game's date, teams, and scores
```

*Input:* Season to look up (like "2023-24")
*Output:* Table of all games with dates, teams, and scores

**`GamesCollector`** - The Game Recorder

*What it does:* Downloads and stores all games for specified seasons.

```python
# Example: Collect all games from recent seasons
from nba_model.data import GamesCollector

collector = GamesCollector()
collector.collect(season_range=["2022-23", "2023-24"])
# Stores: All games from those seasons in the database
```

*Input:* List of seasons to collect
*Output:* Games saved to database

**`PlayByPlayCollector`** - The Play Recorder

*What it does:* Records every single event in a game (shots, rebounds, fouls, timeouts, etc.).

*Why it matters:* Understanding how games flow helps predict future games.

```python
# Example: Get play-by-play for a specific game
collector = PlayByPlayCollector()
collector.collect_game(game_id="0022300123")
# Stores: Every play from that game (might be 400+ events)
```

*Input:* Game ID (a unique number for each game)
*Output:* All plays saved to database

**`ShotsCollector`** - The Shot Tracker

*What it does:* Records where every shot was taken from and whether it went in.

*Why it matters:* Helps understand team and player shooting patterns.

```python
collector = ShotsCollector()
collector.collect_game(game_id="0022300123")
# Stores: Every shot with x,y coordinates, distance, and make/miss
```

*Input:* Game ID
*Output:* Shot data saved to database (typically 80-100 shots per game)

---

### Component 2: Feature Engineering (`nba_model/features/`)

#### What Is It?

The "analytics department" that transforms raw data into meaningful insights.

#### What Does It Do?

Raw data (like "Player X scored 25 points") isn't very useful on its own. This component creates advanced statistics that actually predict winning:

- **RAPM (Regularized Adjusted Plus-Minus):** How much does each player help their team win?
- **Spacing Metrics:** How well does a lineup spread the floor?
- **Fatigue Indicators:** How tired is a team based on travel and schedule?

#### Why Does It Matter?

The difference between good and bad predictions often comes down to using the right features. This is where basketball knowledge meets data science.

#### Key Functions

**`RAPMCalculator`** - The Player Value Calculator

*What it does:* Calculates how many points a player adds (or subtracts) per 100 possessions compared to an average player.

*Think of it like:* A restaurant reviewer who can tell you exactly how much each chef contributes to the restaurant's success, even when they work together.

*The math (simplified):*
- Watch thousands of 2-5 minute "stints" where the same 10 players are on court
- See how the score changes during each stint
- Use statistics to figure out how much each player contributed

```python
from nba_model.features import RAPMCalculator

calculator = RAPMCalculator(
    lambda_=5000,        # How much to smooth the estimates
    min_minutes=100      # Only include players with 100+ minutes
)
results = calculator.fit(stints_data)

# Example output:
# {
#   203507: {'orapm': 4.2, 'drapm': 1.8, 'total_rapm': 6.0},  # LeBron
#   201142: {'orapm': -1.5, 'drapm': 0.3, 'total_rapm': -1.2}  # Average player
# }
```

*Input:* Stint data (who was on court and what happened)
*Output:* Dictionary mapping each player to their RAPM values

*What the output means:*
- `orapm = 4.2`: Team scores 4.2 more points per 100 possessions when this player is on offense
- `drapm = 1.8`: Team allows 1.8 fewer points per 100 possessions when this player is on defense
- `total_rapm = 6.0`: Overall, team is 6 points better per 100 possessions with this player

**`SpacingCalculator`** - The Floor Spacing Analyzer

*What it does:* Measures how well a lineup spreads the basketball court by analyzing where they take shots from.

*Think of it like:* Measuring how well furniture is arranged in a room - spread out is usually better than clustered in a corner.

*The math (simplified):*
- Look at where all 5 players in a lineup typically shoot from
- Draw a shape around those locations (called a "convex hull")
- Measure the area of that shape - bigger = better spacing

```python
from nba_model.features import SpacingCalculator

calculator = SpacingCalculator(min_shots=20)  # Need at least 20 shots
metrics = calculator.calculate_lineup_spacing(
    player_ids=[203507, 2544, 201566, 203954, 1628369],  # 5 players
    shots_df=shots_data
)

# Example output:
# {
#   'hull_area': 850.5,        # Square feet of shooting area
#   'centroid_x': 12.3,        # Average X position
#   'centroid_y': 145.2,       # Average Y position
#   'corner_density': 0.18,    # 18% of shots from corners
#   'shot_count': 245          # Shots analyzed
# }
```

*Input:* List of 5 player IDs and shot data
*Output:* Dictionary of spacing metrics

*What the output means:*
- `hull_area = 850.5`: Lineup shoots from a wide variety of spots (good spacing)
- `corner_density = 0.18`: 18% of shots come from corners (good for three-pointers)

**`FatigueCalculator`** - The Tiredness Tracker

*What it does:* Calculates how fatigued a team might be based on their recent schedule.

*Think of it like:* Knowing that your friend just flew back from Europe and has been working 80-hour weeks - they probably won't perform their best today.

```python
from nba_model.features import FatigueCalculator

calculator = FatigueCalculator()
indicators = calculator.calculate_schedule_flags(
    team_id=1610612747,          # Lakers
    game_date=date(2024, 1, 15),
    games_df=schedule_data
)

# Example output:
# {
#   'rest_days': 1,           # One day since last game
#   'back_to_back': True,     # Second game in two nights
#   '3_in_4': False,          # Not third game in four nights
#   '4_in_5': False,          # Not fourth game in five nights
#   'travel_miles': 2451,     # Miles traveled in last week
#   'home_stand': 0,          # Not on a home stand
#   'road_trip': 3            # Third game of road trip
# }
```

*Input:* Team ID, game date, and schedule data
*Output:* Dictionary of fatigue indicators

*What the output means:*
- `back_to_back = True`: Playing second night in a row (expect fatigue)
- `travel_miles = 2451`: Traveled a lot recently (jet lag effect)
- `road_trip = 3`: On the road for a while (home court advantage gone)

**`EventParser`** - The Play-by-Play Reader

*What it does:* Reads the text descriptions of plays and extracts useful information.

*Think of it like:* Someone who can watch a highlight reel and tell you exactly what type of play happened (fast break, isolation, pick and roll, etc.).

```python
from nba_model.features import EventParser, parse_shot_context

context = parse_shot_context(
    "Curry 26' 3PT Step Back Jump Shot (AST: Green)"
)

# Example output:
# {
#   'distance': 26.0,
#   'shot_type': '3PT',
#   'is_step_back': True,
#   'is_catch_and_shoot': False,
#   'is_transition': False,
#   'assisted_by': 'Green'
# }
```

*Input:* Text description of a play
*Output:* Structured information about what happened

**`SeasonNormalizer`** - The Era Adjuster

*What it does:* Adjusts statistics so they can be compared across different seasons.

*Why it matters:* The NBA has changed dramatically - 100 points was great in 2005 but below average in 2024. This makes stats comparable.

*The math:* Uses z-scores: `adjusted = (raw - season_average) / season_variation`

```python
from nba_model.features import SeasonNormalizer

normalizer = SeasonNormalizer()
normalizer.fit(all_game_stats)  # Learn what's "normal" for each season

# Before: A team with 115 offensive rating in 2015 vs 2024
# After: Both converted to "0.5 standard deviations above average"
normalized_stats = normalizer.transform(game_stats, season="2023-24")
```

*Input:* Raw statistics and season identifier
*Output:* Statistics adjusted for era

---

### Component 3: Machine Learning Models (`nba_model/models/`)

#### What Is It?

The "brain" of the system - neural networks that learn patterns from historical data.

#### What Does It Do?

Three different models work together:
1. **Transformer:** Learns patterns from game flow (sequence of plays)
2. **GNN (Graph Neural Network):** Learns how players interact with each other
3. **Fusion Model:** Combines everything to make final predictions

#### Why Does It Matter?

Traditional statistics miss complex patterns. Machine learning can find relationships that humans would never notice (like "this player performs 8% better against left-handed defenders on back-to-backs").

#### Key Components

**`GameFlowTransformer`** - The Game Flow Reader

*What it does:* Reads sequences of game events (like made shots, turnovers, fouls) and learns patterns that predict outcomes.

*Think of it like:* A film critic who watches movies and learns that certain sequences of scenes predict whether the movie will be good or bad.

*How it works (simplified):*
- Takes the last 50 game events (shots, rebounds, fouls, etc.)
- Uses "attention" to figure out which events matter most
- Produces a summary of the game's "style" or "flow"

```python
from nba_model.models import GameFlowTransformer, EventTokenizer

# Create the model
transformer = GameFlowTransformer(
    vocab_size=15,     # 15 types of events (shot, rebound, foul, etc.)
    d_model=128,       # Size of internal representation
    nhead=4,           # Number of attention heads
    num_layers=2       # Depth of the network
)

# Prepare game events
tokenizer = EventTokenizer(max_seq_len=50)
tokens = tokenizer.tokenize_game(plays_df)

# Get game representation
game_embedding = transformer(tokens)
# Output: 128-dimensional vector representing this game's flow
```

*Input:* Sequence of tokenized game events
*Output:* 128-dimensional vector summarizing the game flow

**`PlayerInteractionGNN`** - The Chemistry Analyzer

*What it does:* Analyzes how the 10 players on court interact with each other.

*Think of it like:* A manager who understands that Team A with Alice, Bob, and Charlie works great, but Bob and Dave clash and hurt productivity.

*How it works (simplified):*
- Creates a graph where each player is a "node"
- Connects players with "edges" (teammates are strongly connected)
- Uses attention to learn which player combinations matter

```python
from nba_model.models import PlayerInteractionGNN, LineupGraphBuilder

# Build the player graph
builder = LineupGraphBuilder(player_features_df)
graph = builder.build_graph(
    home_lineup=[203507, 2544, 201566, 203954, 1628369],
    away_lineup=[201142, 203999, 1628978, 203507, 1629029]
)

# Create and run the GNN
gnn = PlayerInteractionGNN(
    node_features=16,   # Features per player (RAPM, height, etc.)
    hidden_dim=64,
    output_dim=128
)
lineup_embedding = gnn(graph)
# Output: 128-dimensional vector representing lineup interactions
```

*Input:* Graph with 10 players and their features
*Output:* 128-dimensional vector representing lineup chemistry

**`TwoTowerFusion`** - The Decision Maker

*What it does:* Combines all information sources to make final predictions.

*Think of it like:* A CEO who takes reports from different departments (research, analytics, operations) and makes the final decision.

*Architecture:*
- **Tower A:** Processes "static" features (team stats, rest days, RAPM totals)
- **Tower B:** Processes "dynamic" features (Transformer and GNN outputs)
- **Fusion:** Combines both towers and produces three predictions

```python
from nba_model.models import TwoTowerFusion

model = TwoTowerFusion(
    context_dim=32,       # Static features (rest, stats, etc.)
    transformer_dim=128,  # From GameFlowTransformer
    gnn_dim=128,          # From PlayerInteractionGNN
    hidden_dim=256
)

output = model(context_features, transformer_output, gnn_output)

# Output:
# {
#   'win_prob': 0.62,      # 62% chance home team wins
#   'margin': 5.5,         # Predicted home team wins by 5.5 points
#   'total': 223.0         # Predicted total points scored
# }
```

*Input:* Context features + Transformer output + GNN output
*Output:* Three predictions (win probability, margin, total points)

**`FusionTrainer`** - The Teacher

*What it does:* Trains all models by showing them historical games and correcting their mistakes.

*Think of it like:* A teacher who gives students practice tests, grades them, and helps them learn from their mistakes.

```python
from nba_model.models import FusionTrainer, TrainingConfig

config = TrainingConfig(
    learning_rate=0.0001,  # How fast to learn
    batch_size=32,         # Games to process at once
    epochs=50,             # Times to review all data
    patience=10            # Stop if no improvement for 10 epochs
)

trainer = FusionTrainer(transformer, gnn, fusion)
history = trainer.fit(train_loader, val_loader, epochs=50)

# Output: Training history with metrics
# history.best_metrics = {'accuracy': 0.58, 'brier_score': 0.23}
```

*Input:* Training data (historical games with outcomes)
*Output:* Trained models and performance metrics

**`ModelRegistry`** - The Version Keeper

*What it does:* Saves different versions of trained models and tracks their performance.

*Think of it like:* A photo album that keeps different versions of a document, so you can go back to an earlier version if the new one doesn't work.

```python
from nba_model.models import ModelRegistry

registry = ModelRegistry()

# Save a new version
registry.save_model(
    version="1.2.0",
    models={"transformer": t, "gnn": g, "fusion": f},
    metrics={"accuracy": 0.58, "roi": 0.045},
    config={"d_model": 128, "learning_rate": 0.0001}
)

# Load the best version
models = registry.load_model("latest")

# Compare versions
comparison = registry.compare_versions("1.1.0", "1.2.0")
# Shows which version performs better
```

*Input:* Models, metrics, and configuration
*Output:* Saved version that can be loaded later

---

### Component 4: Backtesting (`nba_model/backtest/`)

#### What Is It?

The "quality assurance department" that tests whether the system actually works.

#### What Does It Do?

- Simulates betting on historical games
- Measures profit/loss over time
- Calculates risk metrics
- Validates that the model would have made money in the past

#### Why Does It Matter?

A model might look good on paper but fail in practice. Backtesting catches problems before you bet real money.

#### Key Components

**`WalkForwardEngine`** - The Time Machine Tester

*What it does:* Tests the model on historical data, but in a realistic way - only using information that would have been available at the time.

*Think of it like:* Replaying the stock market from 2020, but only letting yourself make decisions based on what you knew at each point in time (no peeking at the future!).

*How it works:*
```
Fold 1: Train on games 1-500, Test on games 501-600
Fold 2: Train on games 1-550, Test on games 551-650
Fold 3: Train on games 1-600, Test on games 601-700
...and so on
```

```python
from nba_model.backtest import WalkForwardEngine, BacktestConfig

config = BacktestConfig(
    min_train_games=500,           # Need 500 games to start training
    validation_window_games=100,   # Test on 100 games at a time
    kelly_fraction=0.25,           # Bet 25% of full Kelly
    max_bet_pct=0.02               # Never bet more than 2% of bankroll
)

engine = WalkForwardEngine()
result = engine.run_backtest(games_df, trainer, config)

# result.metrics contains:
# {
#   'total_return': 0.12,    # 12% total profit
#   'roi': 0.045,            # 4.5% return on investment
#   'win_rate': 0.54,        # Won 54% of bets
#   'max_drawdown': 0.08,    # Worst losing streak was 8%
#   'sharpe_ratio': 1.2      # Good risk-adjusted return
# }
```

*Input:* Historical games and betting configuration
*Output:* Complete backtest results with all metrics

**`KellyCalculator`** - The Bet Sizer

*What it does:* Calculates the optimal bet size based on your edge and the odds.

*Think of it like:* A financial advisor who tells you exactly how much of your savings to invest based on how good the opportunity is.

*The Kelly Formula:*
```
Full Kelly = (probability × odds - 1) / (odds - 1)

Example:
- You think team has 60% chance to win
- Odds are 2.0 (even money, +100)
- Full Kelly = (0.60 × 2.0 - 1) / (2.0 - 1) = 0.20 / 1.0 = 20%

But full Kelly is too aggressive! We use fractional Kelly:
- Fractional Kelly (25%) = 20% × 0.25 = 5% of bankroll
```

```python
from nba_model.backtest import KellyCalculator

calculator = KellyCalculator(
    fraction=0.25,       # Use 25% of full Kelly (safer)
    max_bet_pct=0.02,    # Never bet more than 2%
    min_edge_pct=0.02    # Only bet if edge is at least 2%
)

result = calculator.calculate_bet_size(
    bankroll=10000,      # You have $10,000
    model_prob=0.60,     # Model says 60% chance
    decimal_odds=2.0     # Even money odds
)

# result:
# {
#   'full_kelly': 0.20,      # Full Kelly says 20%
#   'adjusted_kelly': 0.05,  # After fractional: 5%
#   'bet_amount': 200.00,    # Bet $200 (capped at max)
#   'edge': 0.10             # 10% edge over market
# }
```

*Input:* Bankroll, probability, and odds
*Output:* Recommended bet amount

**`DevigCalculator`** - The True Odds Finder

*What it does:* Removes the sportsbook's built-in profit margin ("vig" or "juice") to find the true implied probability.

*Think of it like:* If a store sells something for $110 that cost them $100, the "vig" is the $10 profit. This calculator figures out the real $100 value.

*Example:*
```
Sportsbook offers:
- Team A: -110 (bet $110 to win $100)
- Team B: -110 (bet $110 to win $100)

Implied probabilities: 52.4% + 52.4% = 104.8% (impossible!)
The extra 4.8% is the "vig" - the sportsbook's profit margin

After devigging: 50% + 50% = 100% (true probabilities)
```

```python
from nba_model.backtest import DevigCalculator

calculator = DevigCalculator(method="power")  # Power method is most accurate

fair_probs = calculator.devig_moneyline(
    home_odds=-150,   # Home team -150
    away_odds=+130    # Away team +130
)

# fair_probs:
# {
#   'home_probability': 0.58,  # True 58% chance
#   'away_probability': 0.42   # True 42% chance
# }
```

*Input:* Odds for both sides
*Output:* True probabilities without the vig

**`BacktestMetricsCalculator`** - The Report Card

*What it does:* Calculates comprehensive performance metrics for the backtest.

*Key metrics it calculates:*

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| ROI | Profit per dollar bet | > 3% |
| Win Rate | Percentage of bets won | > 52% |
| Sharpe Ratio | Return divided by risk | > 1.0 |
| Max Drawdown | Worst losing streak | < 15% |
| CLV (Closing Line Value) | Did you beat the closing line? | > 0% |

```python
from nba_model.backtest import BacktestMetricsCalculator

calculator = BacktestMetricsCalculator()
metrics = calculator.calculate_full_metrics(bets, initial_bankroll=10000)

# metrics contains everything you need:
# {
#   'total_return': 0.12,       # 12% profit
#   'roi': 0.045,               # 4.5% per dollar bet
#   'win_rate': 0.54,           # 54% win rate
#   'sharpe_ratio': 1.2,        # Good risk-adjusted
#   'max_drawdown': 0.08,       # 8% worst drawdown
#   'avg_clv': 0.015,           # 1.5% CLV (beating market)
#   'total_bets': 450           # 450 bets placed
# }
```

---

### Component 5: Monitoring (`nba_model/monitor/`)

#### What Is It?

The "health check department" that watches for problems with the model.

#### What Does It Do?

- Detects when the model starts making bad predictions
- Identifies when NBA patterns change (like a new 3-point revolution)
- Triggers automatic retraining when needed
- Manages different versions of the model

#### Why Does It Matter?

The NBA constantly evolves. A model trained in 2020 might not work in 2025. This component catches problems early before they cost money.

#### Key Components

**`DriftDetector`** - The Change Detector

*What it does:* Detects when the data patterns change from what the model was trained on.

*Think of it like:* A doctor who notices that a patient's vital signs are slowly changing from their normal baseline - even if they're still in the "healthy" range, the change itself is a warning sign.

*Types of drift:*
- **Covariate Drift:** The input data changes (teams playing faster, more 3-pointers)
- **Concept Drift:** The relationship between inputs and outcomes changes

```python
from nba_model.monitor import DriftDetector

detector = DriftDetector(
    reference_data=training_data,     # What "normal" looks like
    p_value_threshold=0.05,           # Statistical significance
    psi_threshold=0.2                 # Population stability threshold
)

result = detector.check_drift(recent_games)

# result:
# {
#   'has_drift': True,
#   'features_drifted': ['pace', 'fg3a_rate'],
#   'details': {
#     'pace': {'psi': 0.25, 'p_value': 0.02},      # Significant drift
#     'fg3a_rate': {'psi': 0.22, 'p_value': 0.03}  # Significant drift
#   }
# }
```

*PSI (Population Stability Index) interpretation:*
- PSI < 0.1: No significant change (all good!)
- 0.1 < PSI < 0.2: Moderate change (watch closely)
- PSI > 0.2: Significant change (consider retraining)

**`RetrainingTrigger`** - The Alarm System

*What it does:* Decides when the model needs to be retrained based on multiple signals.

*Trigger conditions:*
1. **Scheduled:** 7+ days since last training
2. **Drift:** Significant data drift detected
3. **Performance:** ROI drops below -5%
4. **Data Volume:** 50+ new games available

```python
from nba_model.monitor import RetrainingTrigger, TriggerContext

trigger = RetrainingTrigger(
    scheduled_interval_days=7,
    min_new_games=50,
    roi_threshold=-0.05
)

context = TriggerContext(
    last_train_date=date(2024, 1, 8),
    drift_detector=detector,
    recent_data=recent_games,
    games_since_training=75
)

result = trigger.evaluate_all_triggers(context)

# result:
# {
#   'should_retrain': True,
#   'reasons': ['Scheduled retraining due', 'Drift detected in pace'],
#   'priority': 'high'
# }
```

**`ModelVersionManager`** - The Time Capsule

*What it does:* Keeps track of every version of the model and allows rollback if a new version performs worse.

*Think of it like:* The "undo" feature in a word processor - if the new version has problems, you can go back to the previous one.

```python
from nba_model.monitor import ModelVersionManager

manager = ModelVersionManager()

# Create new version after retraining
manager.create_version(
    models=trained_models,
    config=training_config,
    metrics=validation_metrics,
    bump="minor"  # v1.1.0 -> v1.2.0
)

# Compare old and new
comparison = manager.compare_versions("v1.1.0", "v1.2.0")
# Shows which is better on test data

# If new version is worse, rollback
if comparison['winner'] == 'v1.1.0':
    manager.rollback("v1.1.0")
```

---

### Component 6: Predictions (`nba_model/predict/`)

#### What Is It?

The "forecasting department" that generates actual predictions for upcoming games.

#### What Does It Do?

- Loads trained models
- Gathers current data for upcoming games
- Generates predictions (win probability, margin, total)
- Adjusts for injuries and late scratches
- Produces betting signals

#### Key Components

**`InferencePipeline`** - The Prediction Machine

*What it does:* Orchestrates the entire prediction process from raw game information to final predictions.

```python
from nba_model.predict import InferencePipeline
from nba_model.models import ModelRegistry
from nba_model.data import session_scope

with session_scope() as session:
    pipeline = InferencePipeline(
        model_registry=ModelRegistry(),
        db_session=session,
        model_version="latest"
    )

    # Predict a single game
    prediction = pipeline.predict_game("0022300123")

    # Predict all games today
    today_predictions = pipeline.predict_today()

    # Predict games on a specific date
    future_predictions = pipeline.predict_date(date(2024, 1, 20))

# Each prediction contains:
# {
#   'game_id': '0022300123',
#   'home_team': 'BOS',
#   'away_team': 'LAL',
#   'home_win_prob': 0.62,
#   'predicted_margin': 5.5,
#   'predicted_total': 223.0,
#   'confidence': 'high',
#   'top_factors': [('Home RAPM advantage', 0.8), ('Rest advantage', 0.3)]
# }
```

**`InjuryAdjuster`** - The Roster Checker

*What it does:* Adjusts predictions based on which players might be missing due to injury.

*The Bayesian approach:*
- Start with prior probabilities from historical data
- Update based on specific circumstances

*Prior play probabilities (from historical data):*
| Status | Chance of Playing |
|--------|-------------------|
| Probable | 93% |
| Questionable | 55% |
| Doubtful | 3% |
| Out | 0% |

```python
from nba_model.predict import InjuryAdjuster

adjuster = InjuryAdjuster(session)

# Get play probability for a specific player
play_prob = adjuster.get_play_probability(
    player_id=203507,           # LeBron
    injury_status="questionable",
    injury_type="ankle",
    team_context={"back_to_back": True}  # Makes it less likely he plays
)
# Returns: 0.47 (55% prior reduced due to back-to-back)

# Adjust entire prediction
adjusted_prediction = adjuster.adjust_prediction(
    base_prediction=original_prediction,
    injury_report=team_injuries
)
# Returns: Prediction with injury-adjusted probabilities
```

**`SignalGenerator`** - The Recommendation Engine

*What it does:* Compares model predictions to market odds and identifies profitable betting opportunities.

```python
from nba_model.predict import SignalGenerator
from nba_model.backtest import DevigCalculator, KellyCalculator

generator = SignalGenerator(
    devig_calculator=DevigCalculator(),
    kelly_calculator=KellyCalculator(fraction=0.25),
    min_edge=0.02,         # Need at least 2% edge to bet
    bankroll=10000.0
)

signals = generator.generate_signals(predictions, market_odds)

# Each signal contains:
# {
#   'game_id': '0022300123',
#   'matchup': 'LAL @ BOS',
#   'bet_type': 'moneyline',
#   'side': 'home',
#   'model_prob': 0.62,          # Model says 62%
#   'market_prob': 0.55,         # Market implies 55%
#   'edge': 0.07,                # 7% edge
#   'kelly_fraction': 0.0176,    # 1.76% of bankroll
#   'recommended_stake_pct': 0.0176,
#   'confidence': 'high'
# }
```

---

### Component 7: Output (`nba_model/output/`)

#### What Is It?

The "communications department" that presents results in a human-readable format.

#### What Does It Do?

- Generates daily prediction reports
- Creates performance tracking dashboards
- Builds a static website for viewing results
- Archives historical predictions

#### Key Components

**`ReportGenerator`** - The Report Writer

*What it does:* Creates structured reports for predictions and performance.

```python
from nba_model.output import ReportGenerator

generator = ReportGenerator()

# Daily predictions report
daily_report = generator.daily_predictions_report(predictions, signals)
# Returns JSON with all games, predictions, and recommended bets

# Performance tracking report
performance_report = generator.performance_report(period="week")
# Returns: accuracy, ROI, CLV, and other metrics for the period

# Model health report
health_report = generator.model_health_report(drift_results, recent_metrics)
# Returns: drift status, retraining recommendations
```

**`ChartGenerator`** - The Visualization Creator

*What it does:* Creates data for charts and graphs.

```python
from nba_model.output import ChartGenerator

generator = ChartGenerator()

# Bankroll over time
bankroll_data = generator.bankroll_chart(bankroll_history, dates)

# Calibration curve (how accurate are the probabilities?)
calibration_data = generator.calibration_chart(predictions, actuals)

# Monthly ROI
roi_data = generator.roi_by_month_chart(bets)
```

**`DashboardBuilder`** - The Website Builder

*What it does:* Creates a static website you can view in your browser.

```python
from nba_model.output import DashboardBuilder

builder = DashboardBuilder(output_dir="docs")

# Build everything
builder.build_full_site()

# Update just today's predictions
builder.update_predictions(predictions, signals)

# Archive today's results
builder.archive_day(date.today())
```

The website structure:
```
docs/
├── index.html         # Main dashboard with summary stats
├── predictions.html   # Today's game predictions
├── history.html       # Historical performance
├── model.html         # Model health and version info
├── api/
│   ├── today.json     # Today's data in JSON
│   └── history/       # Archived daily data
└── assets/
    ├── style.css
    └── charts.js
```

---

## Glossary

Terms you might encounter, explained simply:

| Term | Simple Explanation |
|------|-------------------|
| **API** | A way for computers to talk to each other (like a waiter taking orders between you and the kitchen) |
| **Backtest** | Testing a strategy on historical data to see if it would have worked |
| **Bankroll** | The total money you have available for betting |
| **Brier Score** | Measures how accurate probability predictions are (lower is better) |
| **CLI** | Command Line Interface - the text-based way to run the program |
| **CLV** | Closing Line Value - whether you got better odds than the final line |
| **Covariate Drift** | When the input data patterns change from what the model learned |
| **Decimal Odds** | European odds format (2.0 means bet $1 to win $2 total) |
| **Devig** | Removing the sportsbook's profit margin to find true probabilities |
| **Drawdown** | How much you've lost from your peak bankroll |
| **Edge** | Your advantage over the market (model probability - market probability) |
| **Feature** | A piece of information used to make predictions |
| **GNN** | Graph Neural Network - AI that learns relationships between connected things |
| **Implied Probability** | The probability suggested by betting odds |
| **Kelly Criterion** | Mathematical formula for optimal bet sizing |
| **Moneyline** | Betting on which team will win (not by how much) |
| **RAPM** | Regularized Adjusted Plus-Minus - measures player impact |
| **ROI** | Return on Investment - profit divided by amount bet |
| **Sharpe Ratio** | Return divided by risk (higher is better) |
| **Spread** | Betting on the margin of victory |
| **Stint** | A period of time where the same 10 players are on court |
| **Transformer** | AI that learns patterns in sequences (like reading a book) |
| **Vig/Juice** | The sportsbook's built-in profit margin on odds |
| **Walk-Forward** | Testing method that respects time order (no peeking at the future) |

---

## Getting Help

If something isn't working:

1. **Check the logs:** Look in the `logs/` folder for error messages
2. **Run diagnostics:** `python -m nba_model.cli data status`
3. **Report issues:** https://github.com/anthropics/claude-code/issues

---

*This documentation was created to help everyone understand the NBA betting model, regardless of technical background. If you have questions or suggestions for improvements, please open an issue!*
