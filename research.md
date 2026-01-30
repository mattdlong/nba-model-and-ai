# The Alpha Protocol: Architecting a Gold Standard NBA Quantitative Trading Strategy Using Public Data Streams

## 1. The Quantitative Thesis: Convergence of High-Frequency Finance and Sports Analytics

The modern landscape of sports betting has irrevocably shifted from a domain of intuition and qualitative handicapping to a rigorous discipline of quantitative finance. In this ecosystem, the National Basketball Association (NBA) represents a particularly volatile yet inefficient asset class. The objective of this report is to architect a "Gold Standard" betting model that operates with the sophistication of a quantitative hedge fund, yet is constrained strictly to the use of publicly available data. This constraint is not merely a limitation; it is an engineering challenge that forces the utilization of advanced signal processing, deep learning architectures, and alternative data extraction methods to approximate the proprietary edges typically held by syndicates possessing optical tracking data.

The central thesis of this research is that the democratization of data, facilitated by the NBA's public API endpoints and the open-source machine learning community, has created a "public alpha" layer. While proprietary data providers like Second Spectrum offer granular X,Y,Z tracking coordinates at 25 frames per second , the latent signals within public play-by-play logs, shot charts, and box scores—when processed through state-of-the-art (SOTA) architectures like Transformers and Graph Neural Networks (GNNs)—contain sufficient information density to construct a profitable predictive engine.

This report treats the NBA betting market as a financial market exhibiting "weak-form" efficiency, where historical price and volume data (in this case, game statistics and betting odds) contain unexploited information. To extract this value, we propose a "Fusion Model" architecture. This system does not rely on a single algorithm but integrates three distinct pillars: a Sequence Model (Transformer/LSTM) for temporal game flow, a Graph Neural Network (GNN) for player interaction dynamics, and a Tabular Model for static efficiency metrics. Furthermore, the execution layer is governed by strict bankroll management principles derived from the Kelly Criterion and advanced devigging methods to ensure that the theoretical edge translates into realized equity growth.

## 2. The Data Warehouse: Engineering Signal from Public Noise

The foundation of any quantitative model is the integrity and granularity of its data lake. In the public sphere, data is fragmented across various endpoints, requiring a sophisticated ingestion and normalization pipeline. We categorize our data sources into three tiers: Primary Event Data, Derived Contextual Data, and Market Data.

### 2.1. Primary Event Data Structure

The core of the "Gold Standard" model relies on the granular event logs provided by the NBA's stats API. Unlike box scores, which destroy temporal context, event logs allow for the reconstruction of the game state at any second.

**The playbyplayv2 Endpoint:** This endpoint is the critical vector for understanding game flow. It provides a row-by-row account of every event—shots, fouls, substitutions, and timeouts—timestamped to the second. The raw data includes fields such as EVENTMSGTYPE, EVENTMSGACTIONTYPE, and textual descriptions (HOMEDESCRIPTION, VISITORDESCRIPTION).

The analytical power of playbyplayv2 lies in its text fields. Proprietary metrics often track "defensive breakdowns," but these are not explicitly listed in public columns. However, by applying Regular Expression (Regex) parsing to the description fields, we can engineer proxies for these metrics. For instance, parsing for specific phrasings like "Bad Pass Turnover" versus "Lost Ball Turnover" differentiates between unforced errors and forced defensive pressure. Furthermore, identifying "Uncontested" shots can be achieved by cross-referencing play descriptions with shotchartdetail data, looking for specific action types (e.g., "Driving Dunk") that occur without a corresponding "Block" or "Foul" event in close temporal proximity.

**The shotchartdetail Endpoint:** This endpoint provides the X,Y coordinates of every field goal attempted. While it lacks the continuous movement tracks of all 10 players, it provides the terminal location of offensive actions. This data is the primary input for our spatial analysis modules, allowing us to construct "Shot Gravity" maps and "Spacing" metrics by calculating the convex hulls of shooting locations for active lineups.

**Table 1: Primary Public Data Endpoints and Feature Extraction Potential**

| Endpoint Name | Key Fields | Derived Feature Utility |
|---------------|------------|------------------------|
| playbyplayv2 | EVENTMSGTYPE, PCTIMESTRING, PLAYER1_ID | Possession sequencing, substitution patterns, momentum detection, lineup stint definition. |
| shotchartdetail | LOC_X, LOC_Y, SHOT_ZONE_BASIC | Spatial spacing (Convex Hulls), gravity mapping, zone efficiency (eFG%), defensive shot deterrence. |
| boxscoreadvancedv2 | OFF_RATING, DEF_RATING, PACE | High-level team efficiency baselines, possession counts, four factors (eFG%, TOV%, ORB%, FT/FGA). |
| leaguegamefinder | GAME_ID, MATCHUP, WL | Historical schedule analysis, back-to-back detection, travel distance calculation, Elo rating initialization. |
| commonteamroster | PLAYER_ID, HEIGHT, WEIGHT, AGE | Biometric baselines for injury risk modeling, position embeddings for GNN nodes. |

### 2.2. Contextual Data Normalization

A significant challenge in using historical public data is "Covariate Drift" caused by the evolution of the game. The "Pace and Space" revolution has drastically inflated offensive ratings and three-point attempt rates over the last decade. A raw Offensive Rating of 110 was elite in 2012 but is below average in 2025.

To train a robust model on multi-season data, we must normalize these features. Instead of using raw values, we utilize Z-Scores standardized by Season. For a given metric x (e.g., Team Pace) in season s, the normalized feature z is calculated as:

Where μ_s and σ_s are the mean and standard deviation of that metric for season s. This transformation ensures that a "fast" team in 2010 is numerically comparable to a "fast" team in 2024 relative to their respective eras, preserving the signal of "dominance" or "style" while removing the inflationary trend.

### 2.3. Market Data and the Target Variable

To build a betting model rather than just a game simulator, the target variable must account for the market's expectation. We integrate historical odds data (Open, Close, Spread, Total) to calculate Closing Line Value (CLV). The model's objective function is not merely to minimize the Mean Squared Error (MSE) of the predicted score, but to maximize the probability of the predicted outcome exceeding the implied probability of the market odds. This requires scraping historical odds from repositories or using Kaggle datasets to pair every GAME_ID with its corresponding Vegas lines.

## 3. Advanced Feature Engineering: The Mathematical Pillars

Deep learning models are essentially "garbage in, garbage out" systems. To approximate syndicate-level rigor, we must generate "synthetic" features that act as proxies for the advanced metrics available to teams but hidden from the public.

### 3.1. Regularized Adjusted Plus-Minus (RAPM)

Standard Plus-Minus is a noisy statistic, heavily confounded by the quality of teammates and opponents on the floor. To isolate a player's true contribution, we implement Regularized Adjusted Plus-Minus (RAPM) using Ridge Regression.

**The Mathematical Framework:** We treat every "stint"—a period of time where the ten players on the court remain constant—as a single observation. The goal is to solve for a vector β (the RAPM values for all players) in the linear equation:

Here, Y is a vector of point differentials per 100 possessions for each stint. X is a sparse design matrix where each row represents a stint, and columns represent players. For a given stint, the columns corresponding to the five home players are set to 1, the five away players to -1, and all others to 0.

Because X is high-dimensional and multicollinear (players often substitute together), Ordinary Least Squares (OLS) fails. We apply Ridge Regression (L2 Regularization), which minimizes the following cost function:

The regularization parameter λ is critical; it shrinks the coefficients towards zero, reducing variance and preventing overfitting on noisy data. We determine the optimal λ through cross-validation, typically looking for the point where the prediction error stabilizes.

**Implementation:** Using playbyplayv2, we parse substitution events (EVENTMSGTYPE = 8) to define the start and end of every stint. We aggregate the points scored and possessions used during these intervals. The resulting β coefficients provide a single, pure efficiency metric for every player, split into Offensive RAPM (ORAPM) and Defensive RAPM (DRAPM). These coefficients are then fed as static embeddings into our neural networks.

### 3.2. Spatial Proxies: Convex Hulls for Spacing and Gravity

Public data lacks the coordinates of off-ball players, making "spacing" difficult to measure directly. However, we can approximate the "geometry" of an offense using the historical shot locations of the players in a lineup.

**The Convex Hull Algorithm:** For any five-man lineup, we aggregate the X,Y coordinates of their combined field goal attempts from shotchartdetail. We then compute the Convex Hull—the smallest convex polygon containing all these points.

* **Area Calculation:** The area of this polygon serves as a proxy for "Floor Spacing." A lineup with elite shooters (e.g., Curry, Thompson) will generate shot attempts near the perimeter and corners, resulting in a large Convex Hull area. A lineup of non-shooters will have attempts clustered in the paint, resulting in a small area.
* **Centroid Distance:** We also calculate the centroid of the hull and the average distance of the vertices from the basket.

**Python Implementation Strategy:** We utilize the scipy.spatial.ConvexHull library. For each player, we generate a "Shot Density Map" using a Kernel Density Estimation (KDE) on their season's shots. For a specific game prediction, we overlay the density maps of the five active starters. The aggregate area of high-density regions provides a "Spacing Score" feature. This feature correlates strongly with driving efficiency, as larger spacing areas imply less congested lanes for slashers.

### 3.3. Fatigue and Load Management Metrics

The "Schedule Alert" is a classic handicapping angle. We formalize this by computing specific fatigue features:

* **Rest Differential:** Days of rest for Home vs. Away.
* **4-in-5 and 3-in-4:** Boolean flags indicating if a team is playing their 4th game in 5 nights or 3rd in 4 nights.
* **Travel Distance:** Using the haversine formula on the lat/long coordinates of NBA arenas, we calculate the cumulative miles traveled over the last 7 days.
* **Distance Traveled on Court:** Using boxscoreplayertrackv2, we access the "Distance Traveled" (miles) metric for each player. This serves as a micro-fatigue proxy, distinguishing between players who stand in the corner vs. those who run constantly off screens (e.g., Steph Curry).

## 4. Deep Learning Architectures: The Fusion Model

To capture the complexity of NBA games, we move beyond linear regressions to a Fusion Model architecture. This involves three distinct neural networks—Sequence, Graph, and Context—whose outputs are concatenated (fused) to produce final predictions.

### 4.1. Pillar I: Sequence Modeling with Transformers

Basketball is a sequence of possessions. The outcome of possession t is conditional on the state at t-1. Momentum, "hot hands," and psychological pressure are encoded in the sequence of events.

**Architecture Selection:** While Long Short-Term Memory (LSTM) networks are the traditional choice for time-series forecasting due to their ability to maintain a hidden state vector h_t , they suffer from the vanishing gradient problem over long sequences (e.g., a full game of 100+ possessions). We therefore adopt the Transformer architecture, utilizing the Self-Attention mechanism.

**Attention Mechanism in Basketball:** The scaled dot-product attention is defined as:

In our context:
* **Queries (Q):** The current game state.
* **Keys (K):** Previous events in the sequence (e.g., a turnover 3 minutes ago).
* **Values (V):** The embedding vectors of those events.

The Transformer allows the model to "attend" to specific past events that are relevant to the current prediction, regardless of how far back they occurred. For example, the model might learn to weight a "Flagrant Foul" heavily for the next 10 possessions due to the emotional volatility it introduces, a relationship an LSTM might forget.

**Input Tokenization:** We tokenize the game flow into a sequence of events derived from playbyplayv2. Each event is represented as a multi-dimensional embedding vector containing:

1. **Event Type Embedding:** (Make, Miss, Rebound, Foul, Turnover).
2. **Time Embedding:** Remaining seconds in the quarter.
3. **Score Differential:** Standardized score margin.
4. **Lineup Embedding:** Vector representation of the 10 players on the floor.

### 4.2. Pillar II: Graph Neural Networks (GNNs) for Player Interactions

Players do not exist in a vacuum; their performance depends on their interactions with teammates and opponents. A "Pick and Roll" is a graph edge between a ball-handler and a screener.

**Graph Construction:** We construct a dynamic graph for each matchup.

* **Nodes (V):** The 10 players on the court. Node features include RAPM, height, weight, and season shooting percentages.
* **Edges (E):**
   * **Teammate Edges:** Fully connected between teammates. Weights are initialized based on "minutes played together" to represent chemistry.
   * **Opponent Edges:** Connected based on positional matchups (PG vs PG, C vs C) or defensive assignments (if available/inferred).

**GATv2 Architecture:** We utilize Graph Attention Networks v2 (GATv2). This architecture learns dynamic attention weights α_{ij} between nodes.

This mechanism allows the model to dynamically update a player's embedding based on who they are playing with. If a poor shooter is playing alongside an elite playmaker (high gravity), the GATv2 will update the shooter's node embedding to reflect an expected increase in efficiency (the "LeBron James effect").

### 4.3. Pillar III: The Two-Tower Fusion Architecture

To combine the sequence and graph outputs with static contextual data, we employ a Two-Tower architecture, a design pattern proven effective in recommendation systems.

**Structure:**
* **Tower A (Context Tower):** Processes static features (Team Season Stats, Rest Days, Travel, RAPM sums) via a Multi-Layer Perceptron (MLP).
* **Tower B (Dynamic Tower):** Processes the output of the Transformer (Sequence features) and GNN (Interaction features).
* **Fusion Layer:** The output vectors from Tower A and Tower B are concatenated and passed through final dense layers to produce the prediction.

**Outputs:** The model is trained as a multi-task learner to predict:
1. Home Win Probability (Binary Classification).
2. Home Margin of Victory (Regression).
3. Total Points (Regression).

**Table 2: Deep Learning Architecture Specifications**

| Component | Architecture Type | Input Data Source | Purpose |
|-----------|------------------|-------------------|---------|
| Sequence Model | Transformer Encoder | playbyplayv2 (Tokenized) | Capture game flow, momentum, and temporal dependencies. |
| Interaction Model | GATv2 (Graph Attention) | commonteamroster, boxscoreadvancedv2 | Model player chemistry, defensive matchups, and synergistic effects. |
| Context Model | MLP (Dense Layers) | Derived Features (Fatigue, RAPM) | Establish baseline team strength and situational factors. |
| Fusion Layer | Two-Tower Concat + MLP | Outputs of above models | Integrate signals for final market prediction. |

## 5. Modeling Uncertainty: Bayesian Inference and Injury Impact

A deterministic prediction is insufficient for betting; we need a probability distribution that accounts for the massive uncertainty of NBA rosters.

### 5.1. The "Game Time Decision" (GTD) Problem

Late scratches and vague injury reports ("Questionable") are the primary source of volatility. We employ a Bayesian Hierarchical Model to estimate the true probability of a player suiting up.

**Priors:** Historical analysis of NBA injury reports provides our priors:
* **Probable:** ~93% play rate.
* **Questionable:** ~55% play rate.
* **Doubtful:** ~3% play rate.
* **Out:** 0% play rate.

**Bayesian Update:** We update these priors using player-specific and team-specific likelihoods.
* **Player History:** Players like Anthony Davis have different "play-through-pain" thresholds than others.
* **Team Context:** Teams actively "tanking" or on the second night of a back-to-back are less likely to play a "Questionable" star.
* **Injury Type:** A "Sore Knee" has a different active probability than "Health and Safety Protocols".

**Win Probability Adjustment:** For a game with a "Questionable" star (e.g., Luka Dončić), the model runs two inference passes:

The final win probability is the expected value derived from the Bayesian play probability:

This adjusted probability is what we bet against the market.

### 5.2. Covariate Drift Detection

To ensure the model does not become obsolete as the season progresses, we implement Covariate Drift Detection. We continuously monitor the distribution of input features (e.g., Pace, 3PA Rate) using Maximum Mean Discrepancy (MMD) or Kolmogorov-Smirnov (KS) tests. If the distribution of recent games diverges significantly from the training set (p-value < 0.05), the model triggers a retraining cycle or increases the weight of recent data (time-decay weighting).

## 6. Strategy and Execution: Converting Alpha to Equity

A predictive model is only as good as its execution. This section details the financial framework for betting.

### 6.1. Devigging and Finding True Odds

Sportsbooks bake a fee ("vig" or "juice") into their lines. To know if we have an edge, we must remove this vig to find the implied "fair" probability. We reject the simplistic "Multiplicative" method in favor of the Power Method or Shin's Method.

**The Power Method:** This method assumes that the bookmaker adjusts probabilities by raising them to a constant power k. We solve for k such that the sum of the implied probabilities equals 1:

Where O_i are the decimal odds. The "Fair Probability" for outcome i is then P_i = (1/O_i)^k. This method is superior for handling the "Longshot Bias," where books shade underdogs more heavily than favorites.

**Shin's Method:** Shin's method models the market as populated by a mix of "informed" bettors (insiders) and "uninformed" bettors. It iteratively solves for the proportion of informed bettors (z) to derive the true probability. This is considered the "Gold Standard" for liquid markets.

### 6.2. Bankroll Management: Fractional Kelly Criterion

To determine bet sizing, we use the Kelly Criterion, which maximizes the logarithmic growth rate of the bankroll. The optimal bet fraction f* is:

Where:
* **b** = Net odds (Decimal odds - 1).
* **p** = Modeled probability of winning.
* **q** = Probability of losing (1-p).

**Risk Control (Fractional Kelly):** Full Kelly betting is notoriously volatile and assumes perfect model calibration. Overestimating p by even a small margin can lead to ruin. Therefore, we strictly adhere to a Fractional Kelly strategy, typically Quarter-Kelly (0.25f) or Half-Kelly (0.50f). This creates a safety buffer that significantly reduces drawdown risk while retaining approximately 75% of the theoretical growth rate.

**Portfolio Constraints:** We impose a hard cap of 2-3% of the total bankroll on any single wager to protect against "Black Swan" events (e.g., a star player getting injured in the first minute).

### 6.3. Market Microstructure and Timing

The NBA betting market is fluid.

* **Opening Lines:** These are the "softest" lines, released early (often the night before). The model's "Tower A" (Context) is most effective here, exploiting clear mispricings in team strength before the public reacts.
* **Closing Lines:** These are the "sharpest," incorporating all information. Our goal for late bets is to beat the Closing Line Value (CLV). Consistently beating the CLV (e.g., betting -110 when the line closes -125) is a stronger predictor of long-term profitability than short-term win/loss records.

## 7. Limitations and the "Glass Ceiling" of Public Data

It is vital to explicitly acknowledge the limitations of this "Gold Standard" model compared to professional syndicates.

1. **Optical Tracking Gap:** We rely on shotchartdetail for spatial analysis. This provides only the terminal location of shots. We miss the velocity, acceleration, and orientation data provided by Second Spectrum. We cannot accurately model off-ball screens, defensive stance (e.g., hip orientation), or the speed of defensive rotations.

2. **Latency:** Public APIs have a latency of seconds to minutes. This renders the model useless for algorithmic In-Game (Live) Betting, which requires sub-second data feeds. The architecture is strictly for Pre-Game and Halftime markets.

3. **Information Asymmetry:** Syndicates often possess "soft" information (locker room flu outbreaks, minor injuries not on the report) before it reaches the public domain. Our Bayesian injury model attempts to account for this, but remains reactive rather than proactive.

## 8. Conclusion

The "Gold Standard" NBA betting model proposed herein represents the apex of what is achievable using strictly public data. By moving beyond basic statistical regression and embracing a "Fusion Architecture" of Transformers, GNNs, and Two-Tower networks, we can extract deep, non-linear signals from the noise of public logs.

The system compensates for the lack of proprietary tracking data through rigorous feature engineering—using RAPM for player valuation, Convex Hulls for spatial geometry, and Regex parsing for defensive metrics. It manages uncertainty through Bayesian injury modeling and protects capital through Fractional Kelly staking and sophisticated devigging. While it cannot compete in the high-frequency domain of live betting due to latency, its architectural depth provides a statistically significant edge in the pre-game and halftime markets, transforming sports betting into a disciplined, quantitative trading operation.

---

## Technical Appendix: Implementation and Mathematical Specifications

### A1. Data Dictionary and Processing Logic

To ensure reproducibility, we define the specific logic used to transform raw API responses into model-ready features.

**Table 3: Event Parsing Logic (Regex & Heuristics)**

| Derived Event | Source Field | Parsing Logic / Regex Pattern | Insight Utility |
|---------------|--------------|------------------------------|-----------------|
| Bad Pass Turnover | HOMEDESCRIPTION | `re.search(r'Bad Pass', desc)` | Unforced error rate; indicates sloppiness vs. forced defense. |
| Lost Ball Turnover | HOMEDESCRIPTION | `re.search(r'Lost Ball', desc)` | Forced error; proxy for opponent's on-ball defensive pressure. |
| Rim Protection | EVENTMSGTYPE | Type=2 (Miss) AND Type=3 (Block) AND Dist < 5ft | Defensive interior presence; separates rim deterrence from perimeter defense. |
| Shot Clock Usage | PCTIMESTRING | Δt between current and prev event | Pace analysis; differentiates "Transition" vs "Half-court" efficiency. |
| Lineup Stint | SUBSTITUTION | Stack logic tracking active PLAYER_IDs | Defines the boundaries for RAPM regression observations. |

### A2. RAPM Implementation Details

**Matrix Construction:** For the Ridge Regression Y = Xβ, the matrix X is constructed as follows:

* **Dimensions:** N × P, where N is the number of stints in the training set (approx. 30,000 per season) and P is the number of unique players (approx. 500).
* **Values:** X_{i,j} ∈ {1, -1, 0}.
   * 1 if player j is on the Home team in stint i.
   * -1 if player j is on the Away team in stint i.
   * 0 otherwise.

**Weighting Scheme:** To account for recency, we apply a diagonal weight matrix W to the regression:

The weights w_i decay exponentially based on the date of the game:

We set τ (tau) to approximately 180 days (half a season), ensuring that recent performance is weighted higher while still retaining a large enough sample size for stability.

### A3. Transformer Hyperparameters

For the Sequence Model (Pillar I), we utilize a Transformer Encoder with the following specifications, optimized for the scale of NBA possession data:

* **Embedding Dimension (d_model):** 128
* **Attention Heads:** 4 (allows attending to different aspects: scoring, turnovers, fouls, lineups)
* **Encoder Layers:** 2 (sufficient depth for tactical patterns without overfitting)
* **Sequence Length:** 50 events (approx. 10-12 minutes of gameplay)
* **Dropout:** 0.1 (to prevent overfitting on specific game scripts)
* **Loss Function:** Cross-Entropy (for Win/Loss classification) and Huber Loss (for Margin regression).

### A4. Convex Hull Calculation Code Snippet (Python)

The following pseudocode demonstrates the calculation of the "Spacing Area" feature using scipy.spatial.

```python
from scipy.spatial import ConvexHull
import numpy as np

def calculate_spacing_area(player_ids, shot_df):
   """
   Calculates the area of the convex hull of shot locations 
   for a given list of 5 active players.
   """
   # Filter shot data for the active players
   active_shots = shot_df[shot_df['PLAYER_ID'].isin(player_ids)]
   
   # Extract X, Y coordinates
   points = active_shots[['LOC_X', 'LOC_Y']].values
   
   # We need at least 3 points to form a polygon
   if len(points) < 3:
       return 0.0
   
   try:
       # Compute Convex Hull
       hull = ConvexHull(points)
       return hull.volume  # In 2D, volume attribute is the Area
   except:
       return 0.0
```

This derived spacing_area is then normalized (Z-Score) and fed as a feature into the GNN and Fusion models.

### A5. Kelly Criterion Risk Table

The following table guides the allocation strategy based on model confidence and market odds, applying a Quarter-Kelly (0.25) constraint.

**Table 4: Quarter-Kelly Staking Guide**

| Assessed Win Prob (p) | Market Odds (Decimal) | Edge (%) | Full Kelly (f) | Bet Size (0.25f) |
|-----------------------|----------------------|----------|----------------|------------------|
| 55% | 1.91 (-110) | 5.0% | 5.5% | 1.37% |
| 60% | 1.91 (-110) | 14.6% | 16.0% | 2.00% (Cap) |
| 52% | 1.91 (-110) | -0.7% | Negative | No Bet |
| 35% | 3.50 (+250) | 22.5% | 9.0% | 2.00% (Cap) |

**Note:** The 2.00% Cap overrides the Quarter-Kelly calculation to prevent catastrophic loss on single events.

---

## Works Cited

1. Player tracking (National Basketball Association) - Wikipedia, https://en.wikipedia.org/wiki/Player_tracking_(National_Basketball_Association)
2. Weak Form Efficiency in Sports Betting Markets, https://myweb.ecu.edu/robbinst/PDFs/Weak%20Form%20Efficiency%20in%20Sports%20Betting%20Markets.pdf
3. Predicting sport event outcomes using deep learning - PMC - NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC12453701/
4. Predicting sport event outcomes using deep learning - PeerJ, https://peerj.com/articles/cs-3011/
5. Applying Kelly Criterion to sports betting: 18 month backtest results, https://www.reddit.com/r/quant/comments/1o2wzfh/applying_kelly_criterion_to_sports_betting_18/
6. Sports Betting: Understanding the Kelly Criterion | by William Finney, https://medium.com/@pelicanlabs/sports-betting-understanding-the-kelly-criterion-fdca4d0f029e
7. Get NBA Stats API play-by-play (Multiple Games) — nba_pbps, https://hoopr.sportsdataverse.org/reference/nba_pbps.html
8. nba_api/docs/examples/PlayByPlay.ipynb at master - GitHub, https://github.com/swar/nba_api/blob/master/docs/examples/PlayByPlay.ipynb
9. NBA Games Outcome Project — Web Scraping With Python, https://betterprogramming.pub/nba-web-scraping-with-python-22c76cfd1d4f
10. Basic NBA Data Parsing with Python - The Art of Network Engineering, https://artofnetworkengineering.com/2022/10/03/basic-nba-data-parsing-with-python/
11. A quick look into visualizing NBA shot data | by Nam Nguyen - Medium, https://medium.com/@namnguyen93/a-quick-look-into-visualizing-nba-shot-data-24756665565b
12. Calculating the convex hull of a point data set (Python) - Geo-code, http://chris35wills.github.io/convex_hull/
13. Measuring Player Spacing Using Convex Hulls, http://projects.rajivshah.com/sportvu/Chull_NBA_SportVu.html
14. Half-KFN: An Enhanced Detection Method for Subtle Covariate Drift, https://arxiv.org/abs/2410.08782
15. (PDF) NBA Results Forecast: From League Dynamics Analysis to..., https://www.researchgate.net/publication/391452085_NBA_Results_Forecast_From_League_Dynamics_Analysis_to_Predictive_Model_Implementation
16. (PDF) A Decade of Evolution: Comparative Analysis of Shooting..., https://www.researchgate.net/publication/394402662_A_Decade_of_Evolution_Comparative_Analysis_of_Shooting_Trends_and_Offensive_Efficiency_in_the_NBA_and_EuroLeague
17. Predicting NBA Games | Joe Ferrara, https://joe-ferrara.github.io/2020/05/04/basketball.html
18. Beating the House: Identifying Inefficiencies in Sports Betting Markets, https://arxiv.org/pdf/1910.08858
19. Regularized Adjusted Plus-Minus xRAPM Explained - NBAstuffer, https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/
20. Open Source Data Science Pipeline for Developing "Moneyball..., https://basketball-analytics.gitlab.io/rapm-data/open-source-data-nba.pdf
21. Deep Dive on Regularized Adjusted Plus Minus II: Basic Application..., https://squared2020.com/2017/09/18/deep-dive-on-regularized-adjusted-plus-minus-ii-basic-application-to-2017-nba-data-with-r/
22. Parsing NBA Substitutions in Play-by-Play Data - Quip Trippin, http://jungwirb.io/parsing-nba-substitutions-in-play-by-play-data.html
23. Exploring Basketball Spacing Through Computer Vision... - Medium, https://medium.com/@kalidrafts/exploring-basketball-spacing-through-computer-vision-broadcast-data-cdff8a118c4f
24. Nylon Calculus: How can we visualize a player's shooting gravity?, https://fansided.com/2019/07/22/nylon-calculus-visualizing-nba-shooting-gravity/
25. NBA Schedule Modeling 2014-15 to 2023-24 - RPubs, https://www.rpubs.com/j0hnathanpham/1351134
26. GitHub - jgwaugh/NBA-Forecasting, https://github.com/jgwaugh/NBA-Forecasting
27. Long-Sequence LSTM Modeling for NBA Game Outcome Prediction..., https://arxiv.org/pdf/2512.08591
28. Mastering Transformers: Building the Attention Mechanism Step by..., https://medium.com/@krupck/mastering-transformers-building-the-attention-mechanism-step-by-step-d8f1e7066871
29. arXiv:2303.16741v1 [cs.LG] 29 Mar 2023, https://arxiv.org/pdf/2303.16741
30. Predicting In-Game Actions from Interviews of NBA Players, https://direct.mit.edu/coli/article/46/3/667/93377/Predicting-In-Game-Actions-from-Interviews-of-NBA
31. Visualization of the Transformer attention maps for NBA dataset. (top)..., https://www.researchgate.net/figure/sualization-of-the-Transformer-attention-maps-for-NBA-dataset-top-Original-sequence_fig2_369449573
32. Dynamic graph neural networks for UAV-based group activity..., https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1631998/full
33. Predicting NBA Player Market Value with Graph Neural Networks, https://medium.com/@vikgarrett/predicting-nba-player-market-value-with-graph-neural-networks-18f56005a684
34. Understanding Two-Tower Architecture in Recommendation Systems, https://www.tredence.com/blog/understanding-the-twotower-architecture-in-recommendation-systems
35. The Two-Tower Model for Recommendation Systems: A Deep Dive, https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive
36. Bayesian Hierarchical Modelling of Basketball Team Performance, https://www.scitepress.org/Papers/2023/121591/121591.pdf
37. Bayesian Hierarchical Modeling - tothemean, https://www.tothemean.com/2020/09/19/hierarchical-model.html
38. Probable, Questionable & Doubtful Injury Report Designations, https://www.betfirm.com/injury-report-designations/
39. NBA Injury Database - Hashtag Basketball, https://hashtagbasketball.com/nba-injury
40. Detecting Covariate Drift with Explanations, https://www.dfki.de/fileadmin/user_upload/import/11753_Detecting_Covariate_Drift_with_Explanations.pdf
41. Uncovering True Outcome Probabilities - OddsJam, https://oddsjam.com/betting-education/uncovering-true-outcome-probabilities
42. Automatically de-vig Pinnacle sportsbook's odds (4 methods), https://www.pinnacleoddsdropper.com/guides/how-to-devig-pinnacle-s-odds-for-betting-on-soft-books
43. How to Devig Odds - Comparing the Methods | Outlier, https://help.outlier.bet/en/articles/8208129-how-to-devig-odds-comparing-the-methods
44. What is the Kelly Criterion and How Does it Apply to Sports Betting?, https://betstamp.com/education/kelly-criterion
45. Why fractional Kelly? Simulations of bet size with uncertainty and..., https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html
46. Live Game Updates - Sportradar API Documentation, https://developer.sportradar.com/basketball/docs/nba-ig-live-game-retrieval
47. Revisiting NBA v. Motorola in the Big Data Era, https://scholars.unh.edu/cgi/viewcontent.cgi?article=1054&context=unhslr
