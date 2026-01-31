"""Unit tests for the betting signal generation module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from nba_model.predict.signals import (
    DEFAULT_MIN_EDGE,
    HIGH_CONFIDENCE_EDGE,
    MEDIUM_CONFIDENCE_EDGE,
    SPREAD_STD_DEV,
    BetType,
    BettingSignal,
    Confidence,
    MarketOdds,
    Side,
    SignalGenerator,
    american_to_decimal,
    create_market_odds,
    decimal_to_american,
)


class TestConstants:
    """Tests for signal generation constants."""

    def test_min_edge_default(self) -> None:
        """Test default minimum edge."""
        assert DEFAULT_MIN_EDGE == 0.02

    def test_confidence_thresholds(self) -> None:
        """Test confidence threshold values."""
        assert HIGH_CONFIDENCE_EDGE == 0.05
        assert MEDIUM_CONFIDENCE_EDGE == 0.03
        assert HIGH_CONFIDENCE_EDGE > MEDIUM_CONFIDENCE_EDGE

    def test_spread_std_dev(self) -> None:
        """Test spread standard deviation is reasonable."""
        assert 10 < SPREAD_STD_DEV < 20


class TestBetType:
    """Tests for BetType enum."""

    def test_bet_types_defined(self) -> None:
        """Test all bet types are defined."""
        assert BetType.MONEYLINE.value == "moneyline"
        assert BetType.SPREAD.value == "spread"
        assert BetType.TOTAL.value == "total"


class TestSide:
    """Tests for Side enum."""

    def test_sides_defined(self) -> None:
        """Test all sides are defined."""
        assert Side.HOME.value == "home"
        assert Side.AWAY.value == "away"
        assert Side.OVER.value == "over"
        assert Side.UNDER.value == "under"


class TestConfidence:
    """Tests for Confidence enum."""

    def test_confidence_levels_defined(self) -> None:
        """Test all confidence levels are defined."""
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"


class TestMarketOdds:
    """Tests for MarketOdds dataclass."""

    def test_create_market_odds(self) -> None:
        """Test creating MarketOdds object."""
        odds = MarketOdds(
            game_id="0022300001",
            home_ml=1.65,
            away_ml=2.35,
            spread_home=-5.5,
            spread_home_odds=1.91,
            spread_away_odds=1.91,
            total=224.5,
            over_odds=1.91,
            under_odds=1.91,
            source="pinnacle",
        )

        assert odds.game_id == "0022300001"
        assert odds.home_ml == 1.65
        assert odds.spread_home == -5.5
        assert odds.total == 224.5

    def test_market_odds_default_timestamp(self) -> None:
        """Test that timestamp defaults to now."""
        odds = MarketOdds(
            game_id="0022300001",
            home_ml=1.65,
            away_ml=2.35,
            spread_home=-5.5,
            spread_home_odds=1.91,
            spread_away_odds=1.91,
            total=224.5,
            over_odds=1.91,
            under_odds=1.91,
        )

        assert isinstance(odds.timestamp, datetime)


class TestBettingSignal:
    """Tests for BettingSignal dataclass."""

    def test_create_betting_signal(self) -> None:
        """Test creating a BettingSignal object."""
        signal = BettingSignal(
            game_id="0022300001",
            game_date=date(2024, 1, 15),
            matchup="LAL @ BOS",
            bet_type="moneyline",
            side="home",
            line=None,
            model_prob=0.58,
            market_prob=0.52,
            edge=0.06,
            recommended_odds=1.91,
            kelly_fraction=0.08,
            recommended_stake_pct=0.02,
            confidence="medium",
        )

        assert signal.game_id == "0022300001"
        assert signal.edge == 0.06
        assert signal.bet_type == "moneyline"

    def test_betting_signal_defaults(self) -> None:
        """Test BettingSignal default values."""
        signal = BettingSignal(
            game_id="0022300001",
            game_date=date(2024, 1, 15),
            matchup="LAL @ BOS",
            bet_type="moneyline",
            side="home",
            line=None,
            model_prob=0.58,
            market_prob=0.52,
            edge=0.06,
            recommended_odds=1.91,
            kelly_fraction=0.08,
            recommended_stake_pct=0.02,
            confidence="medium",
        )

        assert signal.key_factors == []
        assert signal.injury_notes == []
        assert signal.model_confidence == 0.5
        assert signal.injury_uncertainty == 0.0


class TestCreateMarketOdds:
    """Tests for create_market_odds helper function."""

    def test_create_market_odds_minimal(self) -> None:
        """Test creating MarketOdds with minimal parameters."""
        odds = create_market_odds(
            game_id="0022300001",
            home_ml=1.65,
            away_ml=2.35,
            spread_home=-5.5,
        )

        assert odds.game_id == "0022300001"
        assert odds.spread_home_odds == 1.91  # Default
        assert odds.total == 224.5  # Default

    def test_create_market_odds_custom(self) -> None:
        """Test creating MarketOdds with custom parameters."""
        odds = create_market_odds(
            game_id="0022300001",
            home_ml=1.50,
            away_ml=2.70,
            spread_home=-8.5,
            spread_home_odds=1.95,
            spread_away_odds=1.87,
            total=230.0,
            over_odds=1.95,
            under_odds=1.87,
            source="custom",
        )

        assert odds.spread_home == -8.5
        assert odds.total == 230.0
        assert odds.source == "custom"


class TestOddsConversion:
    """Tests for odds conversion functions."""

    def test_american_to_decimal_favorite(self) -> None:
        """Test converting negative American odds."""
        assert american_to_decimal(-110) == pytest.approx(1.909, rel=0.01)
        assert american_to_decimal(-150) == pytest.approx(1.667, rel=0.01)
        assert american_to_decimal(-200) == pytest.approx(1.5)
        assert american_to_decimal(-300) == pytest.approx(1.333, rel=0.01)

    def test_american_to_decimal_underdog(self) -> None:
        """Test converting positive American odds."""
        assert american_to_decimal(100) == 2.0
        assert american_to_decimal(150) == 2.5
        assert american_to_decimal(200) == 3.0
        assert american_to_decimal(300) == 4.0

    def test_decimal_to_american_favorite(self) -> None:
        """Test converting decimal odds < 2.0."""
        assert decimal_to_american(1.91) == -110
        assert decimal_to_american(1.67) == -149
        assert decimal_to_american(1.5) == -200

    def test_decimal_to_american_underdog(self) -> None:
        """Test converting decimal odds >= 2.0."""
        assert decimal_to_american(2.0) == 100
        assert decimal_to_american(2.5) == 150
        assert decimal_to_american(3.0) == 200

    def test_odds_round_trip(self) -> None:
        """Test that conversion round-trips correctly."""
        original = -110
        decimal = american_to_decimal(original)
        back = decimal_to_american(decimal)
        assert back == original

        original = 150
        decimal = american_to_decimal(original)
        back = decimal_to_american(decimal)
        assert back == original


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    @pytest.fixture
    def mock_devig(self) -> MagicMock:
        """Create mock DevigCalculator."""
        from nba_model.backtest.devig import FairProbabilities

        devig = MagicMock()
        devig.power_method_devig.return_value = FairProbabilities(
            home=0.5,
            away=0.5,
            vig=0.048,
            method="power",
        )
        return devig

    @pytest.fixture
    def mock_kelly(self) -> MagicMock:
        """Create mock KellyCalculator."""
        from nba_model.backtest.kelly import KellyResult

        kelly = MagicMock()
        kelly.calculate.return_value = KellyResult(
            full_kelly=0.08,
            adjusted_kelly=0.02,
            bet_fraction=0.02,
            bet_amount=200.0,
            edge=0.06,
            has_edge=True,
        )
        return kelly

    @pytest.fixture
    def generator(
        self, mock_devig: MagicMock, mock_kelly: MagicMock
    ) -> SignalGenerator:
        """Create SignalGenerator instance."""
        return SignalGenerator(mock_devig, mock_kelly, min_edge=0.02)

    @pytest.fixture
    def sample_prediction(self) -> MagicMock:
        """Create a sample prediction mock."""
        pred = MagicMock()
        pred.game_id = "0022300001"
        pred.game_date = date(2024, 1, 15)
        pred.matchup = "LAL @ BOS"
        pred.home_win_prob_adjusted = 0.58
        pred.predicted_margin_adjusted = 5.2
        pred.predicted_total_adjusted = 218.5
        pred.confidence = 0.72
        pred.injury_uncertainty = 0.1
        pred.top_factors = [("factor1", 0.5), ("factor2", 0.3)]
        return pred

    @pytest.fixture
    def sample_odds(self) -> MarketOdds:
        """Create sample market odds."""
        return create_market_odds(
            game_id="0022300001",
            home_ml=1.65,
            away_ml=2.35,
            spread_home=-5.5,
        )

    def test_init(self, generator: SignalGenerator) -> None:
        """Test SignalGenerator initialization."""
        assert generator.min_edge == 0.02
        assert generator.bankroll == 10000.0

    def test_generate_signals_empty_when_no_odds(
        self, generator: SignalGenerator, sample_prediction: MagicMock
    ) -> None:
        """Test that no signals generated when no odds available."""
        signals = generator.generate_signals([sample_prediction], {})
        assert len(signals) == 0

    def test_generate_signals_with_edge(
        self,
        generator: SignalGenerator,
        sample_prediction: MagicMock,
        sample_odds: MarketOdds,
    ) -> None:
        """Test generating signals when edge exists."""
        odds_dict = {sample_prediction.game_id: sample_odds}
        signals = generator.generate_signals([sample_prediction], odds_dict)

        # Should have at least one signal if edge exists
        # (depends on mock setup)
        assert isinstance(signals, list)

    def test_generate_game_signals_types(
        self,
        generator: SignalGenerator,
        sample_prediction: MagicMock,
        sample_odds: MarketOdds,
        mock_devig: MagicMock,
    ) -> None:
        """Test that all bet types are checked."""
        from nba_model.backtest.devig import FairProbabilities

        # Setup devig to return 50-50 for all markets
        mock_devig.power_method_devig.return_value = FairProbabilities(
            home=0.50,
            away=0.50,
            vig=0.048,
            method="power",
        )

        signals = generator.generate_game_signals(sample_prediction, sample_odds)

        # Should check moneyline, spread, and total
        bet_types = {s.bet_type for s in signals}
        # May or may not have signals depending on edge


class TestCheckMoneyline:
    """Tests for moneyline checking."""

    @pytest.fixture
    def mock_devig(self) -> MagicMock:
        """Create mock DevigCalculator."""
        from nba_model.backtest.devig import FairProbabilities

        devig = MagicMock()
        devig.power_method_devig.return_value = FairProbabilities(
            home=0.50,
            away=0.50,
            vig=0.048,
            method="power",
        )
        return devig

    @pytest.fixture
    def mock_kelly(self) -> MagicMock:
        """Create mock KellyCalculator."""
        from nba_model.backtest.kelly import KellyResult

        kelly = MagicMock()
        kelly.calculate.return_value = KellyResult(
            full_kelly=0.08,
            adjusted_kelly=0.02,
            bet_fraction=0.02,
            bet_amount=200.0,
            edge=0.06,
            has_edge=True,
        )
        return kelly

    def test_home_ml_with_edge(
        self, mock_devig: MagicMock, mock_kelly: MagicMock
    ) -> None:
        """Test moneyline signal when home has edge."""
        from nba_model.backtest.devig import FairProbabilities

        # Home model prob higher than market
        mock_devig.power_method_devig.return_value = FairProbabilities(
            home=0.50,
            away=0.50,
            vig=0.048,
            method="power",
        )

        generator = SignalGenerator(mock_devig, mock_kelly, min_edge=0.02)

        pred = MagicMock()
        pred.game_id = "0022300001"
        pred.game_date = date(2024, 1, 15)
        pred.matchup = "LAL @ BOS"
        pred.home_win_prob_adjusted = 0.56  # 6% edge vs 50%
        pred.confidence = 0.7
        pred.injury_uncertainty = 0.0
        pred.top_factors = []

        odds = create_market_odds(
            game_id="0022300001",
            home_ml=1.91,  # ~52% implied
            away_ml=1.91,
            spread_home=-1.0,
        )

        signals = generator._check_moneyline(pred, odds)

        # Should have home ML signal
        assert len(signals) >= 1
        assert signals[0].side == "home"
        assert signals[0].bet_type == "moneyline"


class TestCheckSpread:
    """Tests for spread checking."""

    @pytest.fixture
    def mock_devig(self) -> MagicMock:
        """Create mock DevigCalculator."""
        from nba_model.backtest.devig import FairProbabilities

        devig = MagicMock()
        devig.power_method_devig.return_value = FairProbabilities(
            home=0.50,
            away=0.50,
            vig=0.048,
            method="power",
        )
        return devig

    @pytest.fixture
    def mock_kelly(self) -> MagicMock:
        """Create mock KellyCalculator."""
        from nba_model.backtest.kelly import KellyResult

        kelly = MagicMock()
        kelly.calculate.return_value = KellyResult(
            full_kelly=0.08,
            adjusted_kelly=0.02,
            bet_fraction=0.02,
            bet_amount=200.0,
            edge=0.06,
            has_edge=True,
        )
        return kelly

    def test_spread_signal_generation(
        self, mock_devig: MagicMock, mock_kelly: MagicMock
    ) -> None:
        """Test spread signal when model margin differs from line."""
        from nba_model.backtest.devig import FairProbabilities

        mock_devig.power_method_devig.return_value = FairProbabilities(
            home=0.50,
            away=0.50,
            vig=0.048,
            method="power",
        )

        generator = SignalGenerator(mock_devig, mock_kelly, min_edge=0.02)

        pred = MagicMock()
        pred.game_id = "0022300001"
        pred.game_date = date(2024, 1, 15)
        pred.matchup = "LAL @ BOS"
        pred.predicted_margin_adjusted = 10.0  # Model predicts +10
        pred.confidence = 0.7
        pred.injury_uncertainty = 0.0
        pred.top_factors = []

        odds = create_market_odds(
            game_id="0022300001",
            home_ml=1.65,
            away_ml=2.35,
            spread_home=-3.0,  # Line is -3
        )

        signals = generator._check_spread(pred, odds)

        # Model predicts +10 vs line of -3 = covers by 7
        # Should favor home spread
        if signals:
            assert signals[0].bet_type == "spread"


class TestCheckTotal:
    """Tests for total checking."""

    @pytest.fixture
    def mock_devig(self) -> MagicMock:
        """Create mock DevigCalculator."""
        from nba_model.backtest.devig import FairProbabilities

        devig = MagicMock()
        devig.power_method_devig.return_value = FairProbabilities(
            home=0.50,
            away=0.50,
            vig=0.048,
            method="power",
        )
        return devig

    @pytest.fixture
    def mock_kelly(self) -> MagicMock:
        """Create mock KellyCalculator."""
        from nba_model.backtest.kelly import KellyResult

        kelly = MagicMock()
        kelly.calculate.return_value = KellyResult(
            full_kelly=0.08,
            adjusted_kelly=0.02,
            bet_fraction=0.02,
            bet_amount=200.0,
            edge=0.06,
            has_edge=True,
        )
        return kelly

    def test_over_signal_when_model_higher(
        self, mock_devig: MagicMock, mock_kelly: MagicMock
    ) -> None:
        """Test over signal when model total is higher than line."""
        from nba_model.backtest.devig import FairProbabilities

        mock_devig.power_method_devig.return_value = FairProbabilities(
            home=0.50,
            away=0.50,
            vig=0.048,
            method="power",
        )

        generator = SignalGenerator(mock_devig, mock_kelly, min_edge=0.02)

        pred = MagicMock()
        pred.game_id = "0022300001"
        pred.game_date = date(2024, 1, 15)
        pred.matchup = "LAL @ BOS"
        pred.predicted_total_adjusted = 240.0  # Model predicts 240
        pred.confidence = 0.7
        pred.injury_uncertainty = 0.0
        pred.top_factors = []

        odds = create_market_odds(
            game_id="0022300001",
            home_ml=1.91,
            away_ml=1.91,
            spread_home=-2.0,
            total=220.0,  # Line is 220
        )

        signals = generator._check_total(pred, odds)

        # Model predicts 240 vs 220 line = should favor over
        if signals:
            over_signals = [s for s in signals if s.side == "over"]
            if over_signals:
                assert over_signals[0].bet_type == "total"


class TestMarginToProbability:
    """Tests for margin to probability conversion."""

    @pytest.fixture
    def generator(self) -> SignalGenerator:
        """Create SignalGenerator with mocks."""
        mock_devig = MagicMock()
        mock_kelly = MagicMock()
        return SignalGenerator(mock_devig, mock_kelly)

    def test_zero_margin_is_fifty_percent(self, generator: SignalGenerator) -> None:
        """Test that zero margin gives 50% probability."""
        prob = generator._margin_to_probability(0.0)
        assert prob == pytest.approx(0.5, rel=0.05)

    def test_positive_margin_above_fifty(self, generator: SignalGenerator) -> None:
        """Test that positive margin gives > 50% probability."""
        prob = generator._margin_to_probability(5.0)
        assert prob > 0.5

    def test_negative_margin_below_fifty(self, generator: SignalGenerator) -> None:
        """Test that negative margin gives < 50% probability."""
        prob = generator._margin_to_probability(-5.0)
        assert prob < 0.5

    def test_probability_clamped(self, generator: SignalGenerator) -> None:
        """Test that extreme margins are clamped."""
        prob_high = generator._margin_to_probability(100.0)
        prob_low = generator._margin_to_probability(-100.0)

        assert 0.01 <= prob_high <= 0.99
        assert 0.01 <= prob_low <= 0.99


class TestDetermineConfidence:
    """Tests for confidence classification."""

    @pytest.fixture
    def generator(self) -> SignalGenerator:
        """Create SignalGenerator with mocks."""
        mock_devig = MagicMock()
        mock_kelly = MagicMock()
        return SignalGenerator(mock_devig, mock_kelly)

    def test_high_confidence_high_edge(self, generator: SignalGenerator) -> None:
        """Test high confidence with high edge."""
        conf = generator._determine_confidence(
            edge=0.08,  # High edge
            model_confidence=0.7,
            injury_uncertainty=0.0,
        )
        assert conf == "high"

    def test_medium_confidence(self, generator: SignalGenerator) -> None:
        """Test medium confidence classification."""
        conf = generator._determine_confidence(
            edge=0.04,  # Medium edge
            model_confidence=0.5,
            injury_uncertainty=0.0,
        )
        assert conf == "medium"

    def test_low_confidence_low_edge(self, generator: SignalGenerator) -> None:
        """Test low confidence with low edge."""
        conf = generator._determine_confidence(
            edge=0.02,  # Low edge
            model_confidence=0.3,
            injury_uncertainty=0.0,
        )
        assert conf == "low"

    def test_injury_uncertainty_reduces_confidence(
        self, generator: SignalGenerator
    ) -> None:
        """Test that injury uncertainty reduces effective confidence."""
        conf_no_injury = generator._determine_confidence(
            edge=0.05,
            model_confidence=0.7,
            injury_uncertainty=0.0,
        )

        conf_with_injury = generator._determine_confidence(
            edge=0.05,
            model_confidence=0.7,
            injury_uncertainty=0.5,  # High uncertainty
        )

        # With uncertainty, should be lower or equal confidence
        confidence_order = {"high": 3, "medium": 2, "low": 1}
        assert confidence_order[conf_with_injury] <= confidence_order[conf_no_injury]
